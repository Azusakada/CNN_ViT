"""Dataset and data-loader utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
}


def build_loaders(
    dataset_name: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    download: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    root = Path(data_root).expanduser()
    generator = torch.Generator().manual_seed(seed)

    if dataset_name in _CIFAR_STATS:
        mean, std, num_classes = _CIFAR_STATS[dataset_name]
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        train_set = cls(root=root, train=True, transform=train_tf, download=download)
        val_set = cls(root=root, train=False, transform=val_tf, download=download)
        class_names = list(train_set.classes)
    elif dataset_name == "imagefolder":
        train_dir, val_dir = root / "train", root / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"ImageFolder expects {train_dir} and {val_dir}; each must contain one subdirectory per class."
            )
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_set = datasets.ImageFolder(train_dir, transform=train_tf)
        val_set = datasets.ImageFolder(val_dir, transform=val_tf)
        if train_set.class_to_idx != val_set.class_to_idx:
            raise ValueError("train and val class folders do not match")
        class_names = list(train_set.classes)
        num_classes = len(class_names)
    else:
        raise ValueError("dataset must be one of: cifar10, cifar100, imagefolder")

    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, drop_last=False, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, num_classes, class_names

