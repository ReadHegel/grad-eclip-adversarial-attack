"""
Train / evaluate binary classifiers on ReadHegel/openai-vit-b16-adv-recognition.

Run from repo root:
    uv run python -m training.train

Dataset layout (see DatasetUtils/prepare_dataset.py):
    image, caption, is_attacked (0=clean, 1=attacked); splits train/test.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.data import build_dataloaders, configure_cache_dirs, print_dataset_summary
from training.metrics import binary_roc_curve, plot_roc_curve
from training.models import ModelName, build_model

# =============================================================================
# Training configuration — edit these defaults
# =============================================================================

MODEL: ModelName = "cnn"  # "cnn" | "clip"
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
FREEZE_CLIP_VISION = False  # True = only train classification head

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # increase (e.g. 4) on GPU nodes with enough /dev/shm
# Images keep native resolution; only patch-grid alignment (see training/preprocess.py).

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("/tmp/adv_recognition_checkpoints")
LOG_EVERY_N_BATCHES = 20

# =============================================================================


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    y_true: np.ndarray | None = None
    y_score: np.ndarray | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_counts(preds: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int, int]:
    preds = preds.view(-1)
    labels = labels.view(-1)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    return tp, tn, fp, fn


def metrics_from_counts(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    loss_sum: float,
    n: int,
    *,
    y_true: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
) -> EpochMetrics:
    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    roc_auc = float("nan")
    if y_true is not None and y_score is not None and len(y_true) > 0:
        _, _, roc_auc = binary_roc_curve(y_true, y_score)

    return EpochMetrics(
        loss=loss_sum / max(n, 1),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        y_true=y_true,
        y_score=y_score,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    train: bool,
    log_every: int = LOG_EVERY_N_BATCHES,
) -> EpochMetrics:
    if train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    n_samples = 0
    tp = tn = fp = fn = 0
    label_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    context = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "eval"

    with context:
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            n_samples += batch_size

            preds = outputs.argmax(dim=1)
            btp, btn, bfp, bfn = confusion_counts(preds, labels)
            tp += btp
            tn += btn
            fp += bfp
            fn += bfn

            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]
            label_chunks.append(labels.detach().cpu().numpy())
            score_chunks.append(probs.cpu().numpy())

            if train and batch_idx % log_every == 0:
                tqdm.write(f"  [{desc}] batch {batch_idx}: loss={loss.item():.4f}")

    y_true = np.concatenate(label_chunks) if label_chunks else None
    y_score = np.concatenate(score_chunks) if score_chunks else None
    return metrics_from_counts(tp, tn, fp, fn, loss_sum, n_samples, y_true=y_true, y_score=y_score)


def metrics_to_dict(metrics: EpochMetrics) -> dict[str, float]:
    """Scalar metrics only (omit score arrays stored for ROC plots)."""
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: EpochMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics_to_dict(metrics),
        },
        path,
    )


def format_metrics(prefix: str, m: EpochMetrics) -> str:
    auc_str = f"{m.roc_auc:.4f}" if not np.isnan(m.roc_auc) else "n/a"
    return (
        f"{prefix} | loss={m.loss:.4f} acc={m.accuracy:.4f} "
        f"prec={m.precision:.4f} rec={m.recall:.4f} f1={m.f1:.4f} auc={auc_str}"
    )


def maybe_plot_roc(
    metrics: EpochMetrics,
    path: Path,
    *,
    title: str,
) -> None:
    if metrics.y_true is None or metrics.y_score is None:
        return
    plot_roc_curve(metrics.y_true, metrics.y_score, path, title=title)


def train_and_evaluate(
    model_name: ModelName,
    *,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device_str: str = DEVICE,
    num_workers: int = NUM_WORKERS,
    plot_roc: bool = False,
) -> EpochMetrics:
    configure_cache_dirs()
    set_seed(SEED)
    device = torch.device(device_str)

    print(f"\n=== Loading dataset (cache under /tmp) ===")
    train_loader, test_loader, dataset_dict = build_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
        clip_model_id=CLIP_MODEL_ID,
    )
    print_dataset_summary(dataset_dict)

    print(f"\n=== Building model: {model_name} ===")
    model = build_model(
        model_name,
        clip_model_id=CLIP_MODEL_ID,
        freeze_clip_vision=FREEZE_CLIP_VISION,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_test_f1 = -1.0
    best_metrics: EpochMetrics | None = None
    run_dir = CHECKPOINT_DIR / f"{model_name}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training for {num_epochs} epochs on {device} ===")
    for epoch in range(1, num_epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_m = run_epoch(model, test_loader, criterion, None, device, train=False)
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  {format_metrics('train', train_m)}")
        print(f"  {format_metrics('test ', test_m)}")

        if plot_roc:
            maybe_plot_roc(
                train_m,
                run_dir / f"roc_train_epoch_{epoch:03d}.png",
                title=f"{model_name} train ROC (epoch {epoch})",
            )
            maybe_plot_roc(
                test_m,
                run_dir / f"roc_test_epoch_{epoch:03d}.png",
                title=f"{model_name} test ROC (epoch {epoch})",
            )
            print(f"  ROC plots: {run_dir}/roc_*_epoch_{epoch:03d}.png")

        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch, test_m)

        if test_m.f1 > best_test_f1:
            best_test_f1 = test_m.f1
            best_metrics = test_m
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, test_m)

    assert best_metrics is not None
    print(f"\n=== Best test split ({model_name}) ===")
    print(f"  {format_metrics('test', best_metrics)}")
    print(f"  Checkpoints: {run_dir}")
    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attack-recognition classifiers.")
    parser.add_argument(
        "--model",
        type=str,
        choices=("cnn", "clip"),
        default=MODEL,
        help="Model architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train both CNN and CLIP sequentially and print comparison.",
    )
    parser.add_argument(
        "--plot-roc",
        action="store_true",
        help="Save ROC curve plots (train + test) under the run checkpoint directory each epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        results: dict[str, EpochMetrics] = {}
        for name in ("cnn", "clip"):
            print("\n" + "=" * 60)
            results[name] = train_and_evaluate(
                name,  # type: ignore[arg-type]
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device_str=args.device,
                num_workers=args.num_workers,
                plot_roc=args.plot_roc,
            )
        print("\n" + "=" * 60)
        print("=== Comparison on test split ===")
        for name, m in results.items():
            print(f"  {name:5s} {format_metrics('', m)}")
        return

    train_and_evaluate(
        args.model,  # type: ignore[arg-type]
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        num_workers=args.num_workers,
        plot_roc=args.plot_roc,
    )


if __name__ == "__main__":
    main()
