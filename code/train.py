"""Train and evaluate one model configuration."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data import build_loaders
from models import MODEL_INFO, build_model, count_parameters
from utils import (
    choose_device,
    dump_json,
    environment_info,
    measure_latency,
    plot_history,
    save_confusion_matrix,
    set_seed,
    write_history,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CNN-ViT image classification experiment")
    p.add_argument("--model", choices=MODEL_INFO, default="hybrid_tiny")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagefolder"], default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, choices=[2, 4], default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--limit-train-batches", type=int, default=0, help="0 means no limit")
    p.add_argument("--limit-val-batches", type=int, default=0, help="0 means no limit")
    return p.parse_args()


def make_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    limit_batches: int,
    num_classes: int,
) -> tuple[float, float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    progress = tqdm(loader, leave=False, desc="train" if training else "valid")
    for batch_idx, (images, targets) in enumerate(progress):
        if limit_batches and batch_idx >= limit_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += predictions.eq(targets).sum().item()
        total += batch_size
        if not training:
            indices = (targets * num_classes + predictions).detach().cpu()
            confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
        progress.set_postfix(loss=f"{total_loss / max(total, 1):.4f}", acc=f"{100 * correct / max(total, 1):.2f}")
    return total_loss / total, 100.0 * correct / total, confusion


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    train_loader, val_loader, num_classes, class_names = build_loaders(
        args.dataset, args.data_root, args.image_size, args.batch_size, args.workers, args.download, args.seed
    )
    model = build_model(args.model, num_classes, args.image_size, args.patch_size, args.dropout).to(device)
    optimizer = make_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    scaler_or_none = scaler if scaler.is_enabled() else None

    run_name = args.run_name or f"{args.dataset}_{args.model}_p{args.patch_size}_s{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "config.json", vars(args))
    dump_json(run_dir / "environment.json", environment_info())

    start_epoch, best_acc = 1, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", -math.inf))

    history: list[dict[str, float]] = []
    final_confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler_or_none, args.limit_train_batches, num_classes
        )
        with torch.inference_mode():
            val_loss, val_acc, final_confusion = run_epoch(
                model, val_loader, criterion, device, None, None, args.limit_val_batches, num_classes
            )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": max(best_acc, val_acc),
            "args": vars(args),
            "class_names": class_names,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, run_dir / "best.pt")
        write_history(run_dir / "metrics.csv", history)
        plot_history(run_dir / "curves.png", history)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train {train_acc:.2f}% | "
            f"val {val_acc:.2f}% | best {best_acc:.2f}%"
        )

    best = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    with torch.inference_mode():
        best_val_loss, best_val_acc, final_confusion = run_epoch(
            model, val_loader, criterion, device, None, None, args.limit_val_batches, num_classes
        )
    latency_ms = measure_latency(model, device, args.image_size)
    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "image_size": args.image_size,
        "patch_size": args.patch_size if args.model.startswith("hybrid") else None,
        "seed": args.seed,
        "parameters": count_parameters(model),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "latency_ms_batch1": latency_ms,
        "training_minutes": (time.perf_counter() - started) / 60.0,
        "device": str(device),
    }
    dump_json(run_dir / "summary.json", summary)
    save_confusion_matrix(run_dir / "confusion_matrix.csv", run_dir / "confusion_matrix.png", final_confusion, class_names)
    print(f"Finished. Results: {run_dir}")
    print(summary)


if __name__ == "__main__":
    main()

