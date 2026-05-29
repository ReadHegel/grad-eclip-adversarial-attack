from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset

from training.preprocess import DEFAULT_CLIP_MODEL_ID, build_preprocess_transform

DATASET_ID = "ReadHegel/openai-vit-b16-adv-recognition"

# All Hugging Face / model caches go under /tmp (small home quota).
HF_HOME = Path("/tmp/hf_home")
HF_DATASETS_CACHE = Path("/tmp/hf_datasets_cache")
TRANSFORMERS_CACHE = Path("/tmp/transformers_cache")


def configure_cache_dirs() -> None:
    """Point Hugging Face and transformers caches to /tmp."""
    for path in (HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)


def load_adv_recognition_hf(split: str | None = None) -> Dataset | DatasetDict:
    """
    Load the adversarial-recognition dataset from Hugging Face.

    Columns (see DatasetUtils/prepare_dataset.py and upload_set.py):
      - image: RGB PIL image (~640×478 clean, ~640×480 attacked after attack keepsize)
      - caption: COCO caption string (metadata only for this task)
      - is_attacked: 0 = clean, 1 = adversarially modified
    Splits: train (4000), test (1000).
    """
    configure_cache_dirs()
    dataset = load_dataset(
        DATASET_ID,
        cache_dir=str(HF_DATASETS_CACHE),
    )
    if split is not None:
        return dataset[split]
    return dataset


def _label_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if hasattr(value, "as_py"):
        return int(value.as_py())
    return int(value)


class AdvRecognitionTorchDataset(TorchDataset):
    """PyTorch wrapper: (tensor, label). Uses attack-aligned preprocessing (no augmentations)."""

    def __init__(
        self,
        hf_split: Dataset,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        model_name: str = "cnn",
        clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
    ) -> None:
        self.hf_split = hf_split
        self.transform = transform or build_preprocess_transform(
            model_name, clip_model_id=clip_model_id
        )

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.hf_split[index]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        label = _label_to_int(row["is_attacked"])
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def build_dataloaders(
    batch_size: int,
    num_workers: int,
    model_name: str = "cnn",
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
) -> tuple[DataLoader, DataLoader, DatasetDict]:
    """Return train_loader, test_loader, and the raw HF DatasetDict."""
    dataset_dict = load_adv_recognition_hf()
    preprocess = build_preprocess_transform(model_name, clip_model_id=clip_model_id)

    train_ds = AdvRecognitionTorchDataset(
        dataset_dict["train"],
        transform=preprocess,
        model_name=model_name,
        clip_model_id=clip_model_id,
    )
    test_ds = AdvRecognitionTorchDataset(
        dataset_dict["test"],
        transform=preprocess,
        model_name=model_name,
        clip_model_id=clip_model_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, dataset_dict


def print_dataset_summary(dataset_dict: DatasetDict) -> None:
    """Print split sizes and label distribution."""
    for split_name, split in dataset_dict.items():
        labels = split["is_attacked"]
        attacked = sum(int(v) for v in labels)
        clean = len(labels) - attacked
        sample = split[0]["image"]
        size = sample.size if isinstance(sample, Image.Image) else "?"
        print(
            f"  {split_name}: n={len(split)} | clean={clean} | attacked={attacked} | "
            f"example_size={size} | columns={split.column_names}"
        )
    attacked_feature = dataset_dict["train"].features.get("is_attacked")
    if isinstance(attacked_feature, ClassLabel):
        print(f"  is_attacked names: {attacked_feature.names}")
