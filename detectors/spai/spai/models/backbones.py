import torch
from torch import nn
import clip


CLIP_MEAN: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class CLIPBackbone(nn.Module):
    def __init__(
        self,
        clip_model: str = "ViT-B/16",
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(clip_model, device=device)
        # self.clip = self.clip.float()
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of images using a CLIP backbone and returns intermediate layers."""
        # Make sure that the parameters of LayerNorm are always in FP32, even during FP16
        # training. Otherwise, it will crash, since clip utilizes a custom LayerNorm that
        # always converts the input to LayerNorm to FP32.
        if self.clip.visual.transformer.resblocks[1].ln_1.weight.dtype != torch.float32:
            for m in self.clip.modules():
                if isinstance(m, clip.model.LayerNorm):
                    m.float()

        self.clip.encode_image(x)
        x = torch.stack([h.output for h in self.hooks], dim=2)[1:, :, :, :]
        x = torch.permute(x, (1, 2, 0, 3))

        return x


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        intermediate_layers: tuple[int, ...] = tuple((i for i in range(12)))
    ) -> None:
        super().__init__()

        # Initialize DINOv2 pretrained model.
        self.dino = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.intermediate_layers: tuple[int, ...] = intermediate_layers
        #print(f"Using DINOv2 model: {dinov2_model} with intermediate layers: {self.intermediate_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x: tuple[torch.Tensor] = self.dino.get_intermediate_layers(x, self.intermediate_layers)
        x: torch.Tensor = torch.stack(x, dim=1)
        x = x.to(input_dtype)
        return x
