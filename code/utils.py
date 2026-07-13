"""Reproducibility, checkpointing and result visualization helpers."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [x["epoch"] for x in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [x["train_loss"] for x in history], label="train")
    axes[0].plot(epochs, [x["val_loss"] for x in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [x["train_acc"] for x in history], label="train")
    axes[1].plot(epochs, [x["val_acc"] for x in history], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(path_csv: Path, path_png: Path, matrix: torch.Tensor, class_names: list[str]) -> None:
    array = matrix.cpu().numpy()
    with path_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, array):
            writer.writerow([name, *row.tolist()])
    if len(class_names) > 30:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.55), max(5, len(class_names) * 0.5)))
    im = ax.imshow(array, cmap="Blues")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    plt.close(fig)


@torch.inference_mode()
def measure_latency(model: torch.nn.Module, device: torch.device, image_size: int, warmup: int = 20, runs: int = 100) -> float:
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

