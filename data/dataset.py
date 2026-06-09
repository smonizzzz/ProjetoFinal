import csv
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE  = 512
NUM_ANGLES  = 3
TRAIN_SUBSETS = ["Spinal-AI2024-subset1", "Spinal-AI2024-subset2",
                 "Spinal-AI2024-subset3", "Spinal-AI2024-subset4"]
TEST_SUBSET   = "Spinal-AI2024-subset5"


def get_transforms(split):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    if split == "train":
        return A.Compose([
            A.CLAHE(clip_limit=3.0, p=0.7),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.CLAHE(clip_limit=3.0, p=1.0),
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def load_csv(csv_path):
    samples = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            name   = row[0].strip()
            angles = [float(row[1]), float(row[2]), float(row[3])]
            samples.append((name, angles))
    return samples


class SpinalDataset(Dataset):
    def __init__(self, root, split="train", seed=42):
        self.transforms = get_transforms(split)
        base = Path(root) / "spinal_ai2024"

        if split == "test":
            # Subset 5 exclusivo para teste
            csv_path = base / "Cobb_test_gt.csv" / "Cobb_spinal-AI2024-test_gt.txt"
            img_base = base / "images" / TEST_SUBSET
            raw      = load_csv(csv_path)
            samples  = [(str(img_base / name), angles)
                        for name, angles in raw
                        if (img_base / name).exists()]
        else:
            # Subsets 1-4 para treino e validação
            csv_path = base / "Cobb_train_gt.csv" / "Cobb_spinal-AI2024-train_gt.txt"
            raw      = load_csv(csv_path)

            # Cada subset tem 4000 imagens — associar bloco de 4000 a cada subset
            samples = []
            for i, subset in enumerate(TRAIN_SUBSETS):
                img_base   = base / "images" / subset
                block      = raw[i * 4000 : (i + 1) * 4000]
                for name, angles in block:
                    img_path = img_base / name
                    if img_path.exists():
                        samples.append((str(img_path), angles))

            # Split 80% treino / 20% validação
            rng     = np.random.default_rng(seed)
            idx     = rng.permutation(len(samples))
            n_train = int(len(samples) * 0.8)

            if split == "train":
                samples = [samples[i] for i in idx[:n_train]]
            else:
                samples = [samples[i] for i in idx[n_train:]]

        self.samples = samples
        print(f"[SpinalDataset] split={split} | {len(self.samples)} amostras")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, angles = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transforms(image=image)["image"]
        return {
            "image":  image,
            "angles": torch.tensor(angles, dtype=torch.float32),
        }


def build_dataloaders(root, batch_size=8, num_workers=4, max_samples=None):
    train_ds = SpinalDataset(root, "train")
    val_ds   = SpinalDataset(root, "val")
    test_ds  = SpinalDataset(root, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
