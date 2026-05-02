"""Unified data loading for MLL and PBC datasets."""

import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile

from .paths import MLL_DIR, PBC_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def safe_loader(path):
    try:
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (0, 0, 0))


class SubsetDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(safe_loader(path)), label


def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def make_lt_samples(samples, labels, num_classes: int, imb_factor: int, seed: int = 42):
    """Construct long-tailed version via exponential decay sampling."""
    rng = np.random.RandomState(seed)
    samples_per_class = [[] for _ in range(num_classes)]
    for s, l in zip(samples, labels):
        samples_per_class[l].append(s)

    class_sizes = [len(c) for c in samples_per_class]
    sorted_idx = np.argsort(class_sizes)[::-1]
    n_max = class_sizes[sorted_idx[0]]
    mu = (1.0 / imb_factor) ** (1.0 / (num_classes - 1))

    target_counts = {}
    for rank, cls_idx in enumerate(sorted_idx):
        target = int(n_max * (mu ** rank))
        target = min(target, len(samples_per_class[cls_idx]))
        target = max(target, 1)
        target_counts[cls_idx] = target

    lt_samples = []
    for cls_idx, target in target_counts.items():
        pool = samples_per_class[cls_idx]
        chosen = rng.choice(len(pool), size=target, replace=False)
        lt_samples.extend([pool[i] for i in chosen])

    return lt_samples, target_counts


def build_dataset(name: str = "mll", variant: str = "original", imb_factor: int = 100,
                  batch_size: int = 32, seed: int = 42, num_workers: int = 4) -> Tuple:
    """
    Build train/val/test loaders + class priors + class names + balanced loader.

    Args:
        name: 'mll' or 'pbc'
        variant: 'original' (use as-is) or 'lt' (apply LT downsampling, only for pbc)
        imb_factor: imbalance factor for LT variant
        batch_size: batch size
        seed: random seed for splits
        num_workers: dataloader workers
    """
    if name == "mll":
        data_dir = MLL_DIR
        num_classes = 21
    elif name == "pbc":
        data_dir = PBC_DIR
        num_classes = 8
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_tf, val_tf = get_transforms()

    base_ds = ImageFolder(root=data_dir, loader=safe_loader)
    class_names = base_ds.classes
    all_samples = base_ds.samples
    all_labels = [s[1] for s in all_samples]

    if variant == "lt":
        all_samples, target_counts = make_lt_samples(
            all_samples, all_labels, num_classes, imb_factor, seed)
        all_labels = [s[1] for s in all_samples]
        print(f"  LT variant (imb={imb_factor}): per-class counts = {target_counts}")

    train_s, temp_s, train_l, temp_l = train_test_split(
        all_samples, all_labels, test_size=0.3, stratify=all_labels, random_state=seed)
    val_s, test_s = train_test_split(
        temp_s, test_size=0.5, stratify=temp_l, random_state=seed)

    counts = np.bincount([s[1] for s in train_s], minlength=num_classes).astype(np.float64)
    priors = torch.tensor((counts + 1) / (len(train_s) + num_classes), dtype=torch.float32)

    train_ds = SubsetDataset(train_s, train_tf)
    val_ds = SubsetDataset(val_s, val_tf)
    test_ds = SubsetDataset(test_s, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    sample_weights = 1.0 / counts[np.array([s[1] for s in train_s])]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_s), replacement=True)
    balanced_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)

    print(f"  Dataset {name}/{variant}: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")
    print(f"  Imbalance ratio: {counts.max()/counts.min():.1f}:1")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "balanced_loader": balanced_loader,
        "priors": priors,
        "class_names": class_names,
        "num_classes": num_classes,
    }
