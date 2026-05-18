from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from src.perturbation import perturbation
from torchvision import transforms
from pathlib import Path
import csv


class TrainingDataset(Dataset):
    def __init__(self, transforms=None, train_csv=None):
        assert train_csv is not None, "train_csv must be provided"
        train_csv = Path(train_csv)
        self.images = []        
        # Read CSV
        with open(train_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row["filepath"]
                label = int(row["class"])
                self.images.append((image_path, label))
        random.shuffle(self.images)

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, target]

class EvaluationDatasetFromCSV(Dataset):
    def __init__(self, csv_file: str, transforms=None, perturb=None):
        """
        Args:
            csv_file: CSV path. Must have columns: image,class,generator,split
        """
        self.images = []

        df = pd.read_csv(csv_file)

        # Only include test split
        df = df[df["split"] == "test"]

        for _, row in df.iterrows():
            image_path = row["filepath"]
            label = int(row["class"])

            if os.path.isfile(image_path):
                self.images.append((image_path, label))

        print(f"Loaded {len(self.images)} images from {csv_file}")

        self.transforms = transforms
        self.perturb = perturb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]

        image = Image.open(image_path).convert("RGB")

        # Resize if too small
        if image.size[0] < 224 or image.size[1] < 224:
            image = transforms.Resize((224, 224))(image)

        if self.transforms is not None and self.perturb is None:
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            if random.random() < 0.5:
                image = self.perturb(image)
            else:
                image = self.transforms(image)

        return image_path, image, target

class EvaluationDatasetFromPath(Dataset):
    def __init__(self, dataset_path, transforms=None, perturb=None):
        # Define subdirectories for real and fake samples
        real_dir = os.path.join(dataset_path, "0_real")
        fake_dir = os.path.join(dataset_path, "1_fake")
        # Verify directories exist
        if not os.path.isdir(real_dir):
            raise FileNotFoundError(f"Missing real directory: {real_dir}")
        if not os.path.isdir(fake_dir):
            raise FileNotFoundError(f"Missing fake directory: {fake_dir}")
        
        # Build dataset lists
        self.real = [
            (os.path.join(real_dir, x), 0)
            for x in os.listdir(real_dir)
            if os.path.isfile(os.path.join(real_dir, x))
        ]

        self.fake = [
            (os.path.join(fake_dir, x), 1)
            for x in os.listdir(fake_dir)
            if os.path.isfile(os.path.join(fake_dir, x))
        ]


        self.images = self.real + self.fake

        self.transforms = transforms
        self.perturb = perturb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if image.size[0] < 224 or image.size[1] < 224:
            image = transforms.Resize((224, 224))(image)
        if self.transforms is not None and self.perturb is None:
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)
            else:
                image = self.transforms(image)
        return [image_path, image, target]