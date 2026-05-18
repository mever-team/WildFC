import torch
import torch.nn as nn
import clip
import open_clip
from transformers import AutoImageProcessor, AutoModel, Blip2Model
from PIL import Image
import torchvision.transforms as transforms


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device
        self.backbone = backbone

        # Load and freeze CLIP
        if self.backbone[0] == 'Blip2':
            self.clip = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            self.hooks = [
                Hook(name, module)
                for name, module in self.clip.vision_model.named_modules()
                if "layer_norm2" in name
            ]
        
        elif self.backbone[0] == 'ViT-H-14':
            self.clip, _, self.preprocess = open_clip.create_model_and_transforms(self.backbone[0], pretrained='laion2b_s32b_b79k')
            self.clip.to(device)
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            self.hooks = [
                Hook(name, module)
                for name, module in self.clip.visual.named_modules()
                if "ln_2" in name
            ]

        elif self.backbone[0] == 'Dinov2':
            self.clip = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            self.hooks = [
                Hook(name, module)
                for name, module in self.clip.named_modules()
                if "norm2" in name
            ]
        elif self.backbone[0] == 'Dinov3-vitl16':
            self.clip = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            self.hooks = [
                Hook(name, module)
                for name, module in self.clip.named_modules()
                if "norm2" in name
            ]

        elif self.backbone[0] == 'OpenCLIP-ViT-L-14':
            self.clip, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            self.hooks = [
                Hook(name, module)
                for name, module in self.clip.visual.named_modules()
                if "ln_2" in name
            ]

        else:
            self.clip, self.preprocess = clip.load(self.backbone[0], device=device)
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
            self.hooks = [
                Hook(name, module)
                for name, module in self.clip.visual.named_modules()
                if "ln_2" in name
            ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(self.backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self, train_layer):
        unfrozen_layers = []
        for name, param in self.named_parameters():
            if any(name.startswith(layer) for layer in train_layer):
                param.requires_grad = True
                unfrozen_layers.append(name)
        
        if unfrozen_layers:
            print("Unfrozen:", ", ".join(unfrozen_layers))
        else:
            print("No layers were unfrozen.")
    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_params / total_params * 100:.2f}%)")
    def forward(self, x):
        x = x.to(self.device)

        with torch.no_grad():
            if self.backbone[0] == 'Blip2':
                self.clip.get_image_features(x)
                g = torch.stack([h.output for h in self.hooks], dim=1)[:, :, 0, :]
            elif self.backbone[0] == 'ViT-H-14':
                self.clip.encode_image(x)
                g = torch.stack([h.output for h in self.hooks], dim=1)[:, :, 0, :]
            elif self.backbone[0] == 'Dinov2':
                self.clip(x)
                g = torch.stack([h.output for h in self.hooks], dim=1)[:, :, 0, :]
            elif self.backbone[0] == 'OpenCLIP-ViT-L-14':
                self.clip.encode_image(x)
                g = torch.stack([h.output for h in self.hooks], dim=1)[:, :, 0, :]
            else:
                self.clip.encode_image(x)
                g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)
        return p, z
