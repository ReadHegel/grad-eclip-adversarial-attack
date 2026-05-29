"""
Quick smoke checks for the training pipeline.

    uv run python -m training.smoke
    uv run python -m training.smoke --clip   # also build CLIP (slow: downloads weights)
"""

from __future__ import annotations

import argparse
import time

import torch

from training.data import build_dataloaders, configure_cache_dirs, print_dataset_summary
from training.models import build_model


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


def run_smoke(*, include_clip: bool = False, device: str | None = None) -> None:
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)
    print(f"device: {device_obj}")

    t0 = time.perf_counter()
    configure_cache_dirs()
    train_loader, test_loader, dataset_dict = build_dataloaders(
        batch_size=2,
        num_workers=0,
        model_name="cnn",
    )
    sample_images, _ = next(iter(train_loader))
    print(f"  tensor shape after preprocess: {tuple(sample_images.shape)}")
    print(f"dataset + dataloaders: {_elapsed(t0)}")
    print_dataset_summary(dataset_dict)

    t1 = time.perf_counter()
    cnn = build_model("cnn").to(device_obj)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device_obj), labels.to(device_obj)
    with torch.no_grad():
        logits = cnn(images)
    print(f"cnn forward batch={images.shape[0]} logits={tuple(logits.shape)}: {_elapsed(t1)}")

    t2 = time.perf_counter()
    cnn.eval()
    with torch.no_grad():
        _ = cnn(images)
    print(f"test batch ({len(test_loader.dataset)} samples, 1 batch): {_elapsed(t2)}")

    if include_clip:
        t3 = time.perf_counter()
        clip_model = build_model("clip").to(device_obj)
        _, clip_loader, _ = build_dataloaders(
            batch_size=2,
            num_workers=0,
            model_name="clip",
        )
        clip_images, _ = next(iter(clip_loader))
        with torch.no_grad():
            _ = clip_model(clip_images.to(device_obj))
        print(f"clip forward: {_elapsed(t3)}")

    print("\nSmoke OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline smoke test.")
    parser.add_argument("--clip", action="store_true", help="Also load CLIP weights and run one batch.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(include_clip=args.clip, device=args.device)
