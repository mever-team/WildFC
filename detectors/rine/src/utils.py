import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
import os
from io import BytesIO
import pickle
import copy
import json
import random
import time
import pandas as pd
import glob
import cv2
import numpy as np
import albumentations as A
from PIL import Image#, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import (
    accuracy_score, 
    average_precision_score, 
    roc_auc_score, 
    confusion_matrix
)
from albumentations.augmentations.transforms import ImageCompression, GaussNoise
from tqdm import tqdm
import shutil
from src.data import TrainingDataset
from src.models import Model
from src.ablations import ModelAblations

from src.texture_crop import texture_crop

def get_transforms(backbone, processing_method, augmentations=True):
    transformations = []
    
    if processing_method == 'centercrop':
        transformations.append(transforms.CenterCrop(224))
        transformations.append(transforms.ToTensor())
    elif processing_method == 'tencrop':
        transformations.append(transforms.TenCrop(224))
        transformations.append(transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])))
    elif processing_method == 'texturecrop':
        transformations.append(transforms.Lambda(lambda img: texture_crop(img))) 
        transformations.append(transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])))
        transformations.append(transforms.Lambda(lambda x: x / 255))

    if backbone == 'Blip2' or backbone == 'Dinov2':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)              
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

    transformations.append(transforms.Normalize(mean=mean, std=std))
    transformations_test = transforms.Compose(transformations)

    if augmentations:
        augmentations_transforms = [
            transforms.Lambda(lambda img: data_augment(img)),
            transforms.RandomHorizontalFlip(p=0.5)]
        transformations = augmentations_transforms + transformations  

    transformations_train = transforms.Compose(transformations)

    return transformations_train, transformations_test


def get_loaders(
    experiment, transforms_train, transforms_val, transforms_test, workers, ds_frac=None
):
    
    
    train = DataLoader(
        TrainingDataset(transforms=transforms_train, train_csv= experiment["training_set_csv"]),
        batch_size=experiment["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    val = None
    
    return train, val


def train_one_experiment(
    experiment,
    model_name,
    epochss,
    epochs_reduce_lr,
    transforms_train,
    transforms_val,
    transforms_test,
    workers,
    device,
    without=None,  # None, contrastive, alpha, intermediate
    store=False,
    ds_frac=None,
):
    seed_everything(0)

    train, val = get_loaders(
        experiment=experiment,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_test=transforms_test,
        workers=workers,
        ds_frac=ds_frac,
    )
    if experiment.get("pretrained", None) is not None:
        model = get_our_trained_model(
            bcb=experiment["backbone"][0],
            model_name=experiment["pretrained"],
            device=device)
    else:
        if without is not None and without in ["alpha", "intermediate"]:
            model = ModelAblations(
                backbone=experiment["backbone"],
                nproj=experiment["nproj"],
                proj_dim=experiment["proj_dim"],
                without=without,
                device=device,
            )
        else:
            model = Model(
                backbone=experiment["backbone"],
                nproj=experiment["nproj"],
                proj_dim=experiment["proj_dim"],
                device=device,
            )
    model.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    if without is None or without != "contrastive":
        supcon = SupConLoss()
        print("Using SupCon Loss with factor:", experiment["factor"])

    print(json.dumps(experiment, indent=2))
    results = {"val_loss": [], "val_acc": [], "test": {}}
    rlr = 0
    training_time = 0
    print("Trainable Parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(max(epochss)):
        training_epoch_start = time.time()
        # Reduce learning rate
        if epoch + 1 in epochs_reduce_lr:
            print( f"\nReducing learning rate from {optimizer.param_groups[0]['lr']} to {experiment['lr'] / 10**(rlr + 1)}" )
            rlr += 1
            optimizer.param_groups[0]["lr"] = experiment["lr"] / 10**rlr

        # Training
        model.train()
        for i, data in enumerate(train):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            b, t, c, h, w = images.size()
            outputs = model(images.view(-1, 3, 224, 224))
            optimizer.zero_grad()
            #print("Training step output/labels:")
            #print(outputs[0].reshape(b, t, *outputs[0].size()[1:]).mean(dim=1).shape, labels.float().view(-1, 1).shape)
            loss_ = bce(outputs[0].reshape(b, t, *outputs[0].size()[1:]).mean(dim=1), labels.float().view(-1, 1))
            if without is None or without != "contrastive":
                print("Using contrastive loss")
                loss_ += experiment["factor"] * supcon(
                    F.normalize(outputs[1].reshape(b, t, *outputs[1].size()[1:]).mean(dim=1)).unsqueeze(1), labels
                )
            loss_.backward()
            optimizer.step()
            print(
                f"\r[Epoch {epoch + 1:02d}/{max(epochss):02d} | Batch {i + 1:04d}/{len(train):04d} | Time {training_time + time.time() - training_epoch_start:1.1f}s] loss: {loss_.item():1.4f}",
                end="",
            )
        training_time += time.time() - training_epoch_start
        if (epoch+1)%10 == 0 or epoch+1 == 1 :
            if store:
                ckpt_name = f"{experiment['save_ckpt_path']}.pth"
                ckpt_name = os.path.join(experiment['save_ckpt_path'], f"{model_name}.pth")
                print(f"Saving {ckpt_name} ...")
                torch.save(
                    {
                        k: model.state_dict()[k]
                        for k in model.state_dict()
                        if "clip" not in k
                    },
                    ckpt_name,
                )
    
def get_our_trained_model(bcb, model_name, device):
    if bcb == "Dinov2":
        nproj = 1
        proj_dim = 512
        model = Model(
            backbone=("Dinov2", 1024),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )
    elif bcb == "ViT-L/14":
        nproj = 2
        proj_dim = 1024
        model = Model(
            backbone=("ViT-L/14", 1024),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )
    elif bcb == "ViT-H-14":
        nproj = 4
        proj_dim = 256
        model = Model(
            backbone=("ViT-H-14", 1280),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )
    elif bcb == "Blip2":
        nproj = 1
        proj_dim = 1408
        model = Model(
            backbone=("Blip2", 1408),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )

    elif bcb == "OpenCLIP-ViT-L-14":
        nproj = 2
        proj_dim = 128
        model = Model(
            backbone=("OpenCLIP-ViT-L-14", 1024),
            nproj=nproj,
            proj_dim=proj_dim,
            device=device,
        )

    setting = "ldm"
    missing_keys = []
    try:
        state_dict = torch.load(f"ckpt/{model_name}.pth", map_location=device)
    except:
        try:
            state_dict = torch.load(model_name, map_location=device)
        except:
            state_dict = torch.load(f"{model_name}.pth", map_location=device)

    for name in state_dict:
        attr_root = name.split(".")[0]
        if not hasattr(model, attr_root):
            missing_keys.append(name)
        
        exec(
            f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
        )
    # --- ADD THIS -------------------------------------------------------
    if missing_keys:
        print("[WARNING] The following keys do NOT exist in the model:")
        for mk in missing_keys:
            print("   •", mk)
    else:
        print("[INFO] All checkpoint keys matched model attributes.")
    print(f'Pretrained_model_path: {model_name}')
    return model


# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def evaluation(model, test, device, processing_method, ours=False, metrics_csv=None, predictions_csv=None, eval_csvs=None):
    accs, aps, aucs = [], [], []
    tprs, fprs = [], []
    log = {}
    csv_rows = []  # ✅ initialize before loop
    
    for g, loader in test:
        model.eval()
        y_true = []
        y_score = []
        
        with torch.no_grad():
            for data in tqdm(loader, desc=f"Processing {g}", leave=False):
                image_paths, images, labels = data
                images, labels = images.to(device), labels.to(device)
                b, t, c, h, w = images.size()
                if ours:
                    outputs = model(
                        images if processing_method == "centercrop" else images.view(-1, 3, 224, 224)
                    )[0]
                else:
                    outputs = model(images)
                if processing_method == "centercrop":
                    probs = torch.sigmoid(outputs).cpu().numpy().tolist()
                else:
                    probs = (
                        torch.sigmoid(outputs.view(images.shape[0], images.shape[1]).mean(1))
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                labels_np = labels.cpu().numpy().tolist()
                y_true.extend(labels_np)
                y_score.extend(probs)

                # 🔹 Collect per-image info for CSV
                for path, prob, label in zip(image_paths, probs, labels_np):
                    csv_rows.append({
                        "filepath": path,
                        "generator": g,
                        "class": int(label),
                        "probability": float(prob),
                    })
        preds = np.array(y_score) > 0.5
        test_acc = accuracy_score(np.array(y_true), preds)
        test_ap = average_precision_score(y_true, y_score)
        test_auc = roc_auc_score(y_true, y_score)
        
        # Confusion matrix for TPR/FPR
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        test_tpr = tp / (tp + fn + 1e-9)  # Recall
        test_fpr = fp / (fp + tn + 1e-9)

        accs.append(test_acc)
        aps.append(test_ap)
        aucs.append(test_auc)
        #tprs.append(test_tpr)
        #fprs.append(test_fpr)

        log[g] = {
            "acc": test_acc,
            "ap": test_ap,
            "auc": test_auc,
            #"tpr": test_tpr,
            #"fpr": test_fpr,
        }

        print(
            f"{g}: AUC: {100*test_auc:1.2f} | "
            f"AP: {100*test_ap:1.2f} | "
            f"ACC: {100*test_acc:1.2f} | "
            #f"TPR: {100*test_tpr:1.2f} | "
            #f"FPR: {100*test_fpr:1.2f}"
        )

    # Mean values
    mean_acc = np.mean(accs)
    mean_ap  = np.mean(aps)
    mean_auc = np.mean(aucs)
    #mean_tpr = np.mean(tprs)
    #mean_fpr = np.mean(fprs)

    #print(
    #    f"Mean: AUC: {100*mean_auc:1.2f} | AP: {100*mean_ap:1.2f} | "
       # f"ACC: {100*mean_acc:1.2f} | TPR: {100*mean_tpr:1.2f} | FPR: {100*mean_fpr:1.2f}"
    #)

    # Save CSV
    if metrics_csv is not None:
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generator", "acc", "ap", "auc"])

            # Per generator
            for g, metrics in log.items():
                writer.writerow([
                    g,
                    metrics["acc"],
                    metrics["ap"],
                    metrics["auc"],
                    #metrics["tpr"],
                   # metrics["fpr"],
                ])

            # Mean row
            writer.writerow([
                "mean",
                mean_acc,
                mean_ap,
                mean_auc,
            ])
    if predictions_csv is not None:
        os.makedirs(os.path.dirname(predictions_csv), exist_ok=True)
        with open(predictions_csv, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "generator", "probability", "class"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved detailed predictions to: {predictions_csv}")

    if eval_csvs is not None:
    # Step 1: Load all original CSVs
        dfs = [pd.read_csv(f) for f in eval_csvs]
        # Step 2: Concatenate them into a single dataframe
        original_df = pd.concat(dfs, ignore_index=True)
        # Step 3: Load predictions.csv
        predictions = pd.read_csv(predictions_csv)

        # Step 5: Merge predictions into original dataframe
        merged = original_df.merge(
            predictions[["filepath", "class", "generator", "probability"]],
            how="left",
            on=["filepath", "class", "generator"]
        )
        # Step 5: Save the updated predictions
        merged.to_csv(predictions_csv, index=False)
def data_augment(img):
    img = np.array(img)

    # JPEG compression
    if random.random() < 0.5:
        method = sample_discrete(["cv2", "pil"])
        qual = sample_discrete([30, 100])
        img = jpeg_from_key(img, qual, method)
    
    # WEBP compression
    if random.random() < 0.5:
        compression_transform = ImageCompression(
            quality_range=(30, 100),  
            compression_type="jpeg", 
            p=1.0)
        augmented = compression_transform(image=img) 
        img = augmented['image']  

    # Gaussian Blur
    if random.random() < 0.5:
        sig = sample_continuous([0.0, 3.0])
        gaussian_blur(img, sig)

    # Gaussian Noise 
    if random.random() < 0.5:
        noise_transform = GaussNoise(var_limit=(10.0, 30.0), p=1.0)  
        augmented = noise_transform(image=img)
        img = augmented['image']

    # Rotation
    if random.random() < 0.5:
        rotation_transform = A.Rotate(limit=(-90, 90), p=1.0)
        augmented = rotation_transform(image=img)
        img = augmented['image']
    
    # Invert
    #if random.random() < 0.5:
    #    img = cv2.bitwise_not(img)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def gather_fp_fn(csv_file, save_path, threshold=0.5, max_samples=50):
    """
    csv_file: path to CSV with columns ['image_id','Probability','Label','Threshold','image_path']
    save_path: folder where FP and FN folders will be created
    threshold: probability threshold to classify as positive
    max_samples: max number of images to copy for FP and FN
    """
    df = pd.read_csv(csv_file)

    fp_folder = os.path.join(save_path, "False_Positive")
    fn_folder = os.path.join(save_path, "False_Negative")
    os.makedirs(fp_folder, exist_ok=True)
    os.makedirs(fn_folder, exist_ok=True)

    false_positives = []
    false_negatives = []

    # Collect FP and FN image paths
    for _, row in df.iterrows():
        img_path = row['image_path']
        prob = row['Probability']
        label = row['Label']
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue

        pred = 1 if prob > threshold else 0

        if pred == 1 and label == 0:
            false_positives.append(img_path)
        elif pred == 0 and label == 1:
            false_negatives.append(img_path)

    # Sample if more than max_samples
    if len(false_positives) > max_samples:
        false_positives = random.sample(false_positives, max_samples)
    if len(false_negatives) > max_samples:
        false_negatives = random.sample(false_negatives, max_samples)

    # Copy selected images
    for path in false_positives:
        shutil.copy(path, fp_folder)
    for path in false_negatives:
        shutil.copy(path, fn_folder)

    print(f"Copied {len(false_positives)} False Positives to: {fp_folder}")
    print(f"Copied {len(false_negatives)} False Negatives to: {fn_folder}")
