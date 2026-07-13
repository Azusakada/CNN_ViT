"""Run the recommended comparison/ablation matrix and aggregate summaries."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagefolder"], default="cifar10")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./runs")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--download", action="store_true")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--quick", action="store_true", help="Two batches and one epoch for pipeline validation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("train.py")
    matrix = [
        ("cnn_tiny", 2, "comparison_cnn"),
        ("vit_tiny", 2, "comparison_vit"),
        ("hybrid_tiny", 2, "comparison_hybrid_p2"),
        ("hybrid_tiny", 4, "ablation_hybrid_p4"),
        ("hybrid_no_fusion", 2, "ablation_no_fusion"),
    ]
    for model, patch, run_name in matrix:
        command = [
            sys.executable,
            str(script),
            "--model",
            model,
            "--patch-size",
            str(patch),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--output-dir",
            args.output_dir,
            "--run-name",
            run_name,
            "--image-size",
            str(args.image_size),
            "--epochs",
            str(1 if args.quick else args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.download:
            command.append("--download")
        command.append("--amp" if args.amp else "--no-amp")
        if args.quick:
            command.extend(["--limit-train-batches", "2", "--limit-val-batches", "2"])
        print("\nRunning:", " ".join(command))
        subprocess.run(command, check=True)

    rows = []
    for _, _, run_name in matrix:
        summary_path = Path(args.output_dir) / run_name / "summary.json"
        rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    fields = [
        "run_name",
        "model",
        "dataset",
        "image_size",
        "patch_size",
        "seed",
        "parameters",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "latency_ms_batch1",
        "training_minutes",
        "device",
    ]
    output = Path(args.output_dir) / "experiment_summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll experiments finished. Summary: {output}")


if __name__ == "__main__":
    main()

