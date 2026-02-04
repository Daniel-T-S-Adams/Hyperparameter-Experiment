from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from .utils import seed_worker


def _mnist_to_tensors(mnist):
    data = mnist.data.float().unsqueeze(1).div(255.0)
    targets = mnist.targets.long()
    return data, targets


def _maybe_preload(mnist, preload: str, device: torch.device):
    data, targets = _mnist_to_tensors(mnist)
    if preload == "cuda" and device.type == "cuda":
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    return TensorDataset(data, targets)


def get_datasets(data_cfg: dict, seed: int, device: torch.device):
    transform = transforms.ToTensor()
    preload = str(data_cfg.get("preload", "none")).lower()

    train_dataset = datasets.MNIST(
        root=data_cfg["root"],
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_cfg["root"],
        train=False,
        download=True,
        transform=transform,
    )

    if preload in {"cpu", "cuda"}:
        train_dataset = _maybe_preload(train_dataset, preload, device)
        test_dataset = _maybe_preload(test_dataset, preload, device)

    train_size = data_cfg["train_size"]
    val_size = data_cfg["val_size"]
    if train_size + val_size != len(train_dataset):
        raise ValueError(
            f"train_size + val_size must equal {len(train_dataset)}; "
            f"got {train_size} + {val_size}."
        )

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )

    return train_subset, val_subset, test_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    training_cfg: dict,
    evaluation_cfg: dict,
    data_cfg: dict,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    persistent_workers = data_cfg.get("persistent_workers", False)
    preload = str(data_cfg.get("preload", "none")).lower()

    if preload == "cuda":
        num_workers = 0
        pin_memory = False
        persistent_workers = False

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        **loader_kwargs,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_cfg["test_batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
