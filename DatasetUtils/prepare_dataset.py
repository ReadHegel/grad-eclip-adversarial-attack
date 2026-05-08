from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

from models import build_clip_model


MODEL_KEYS = [
	"openai-vit-b32",
	"openai-vit-b16",
	"openai-vit-l14",
	"google-siglip2-b32-256",
	"facebook-metaclip2-b16",
]

# --- Simple global config (edit as needed) ---------------------------------
# Which models to generate datasets for (set to a single model to limit work)
SELECT_MODEL_KEYS = ["openai-vit-b16"]

# Sizes and randomness
SEED = 42
TRAIN_SIZE = 4000
TEST_SIZE = 1000
ATTACK_RATIO = 0.5

# Paths
COCO_ROOT = Path("DatasetUtils/data/coco_test_set")
QUICKDRAW_ROOT = Path("DatasetUtils/data/quick_draw_dataset/npy_files")
OUTPUT_ROOT = Path("DatasetUtils/data/generated_datasets")
# Target save size (width, height)
TARGET_SIZE = (640, 478)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePlan:
	sample_id: str
	split: str
	is_attacked: bool
	source_index: int
	source_image_name: str
	caption: str


class QuickDrawSampler:
	def __init__(self, npy_root: Path, seed: int) -> None:
		self.npy_root = npy_root
		self.rng = random.Random(seed)
		self.npy_files = sorted(npy_root.glob("*.npy"))

		if not self.npy_files:
			raise FileNotFoundError(f"No .npy files found in {npy_root}")

	def sample(self) -> tuple[torch.Tensor, Path, int]:
		file_path = self.rng.choice(self.npy_files)
		data = np.load(file_path, mmap_mode="r")
		row_index = self.rng.randrange(data.shape[0])

		raw = np.asarray(data[row_index], dtype=np.uint8)
		if raw.ndim == 1:
			side = 28
			raw = raw.reshape(side, side)

		tensor = torch.from_numpy(raw).float() / 255.0

		return tensor, file_path, row_index


class AdvRecognitionDataset(torch.utils.data.Dataset):
	"""Minimal loader: returns (image PIL, caption str, is_attacked bool)."""

	def __init__(self, root: str | Path, split: str | None = None) -> None:
		self.root = Path(root)
		metadata_path = self.root / "metadata.csv"
		if not metadata_path.exists():
			raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

		with metadata_path.open("r", encoding="utf-8", newline="") as handle:
			self.metadata = list(csv.DictReader(handle))
		if split is not None:
			self.metadata = [row for row in self.metadata if row.get("split") == split]

	def __len__(self) -> int:
		return len(self.metadata)

	def __getitem__(self, index: int) -> dict[str, Any]:
		row = self.metadata[index]
		image_path = self.root / row["image_path"]
		image = Image.open(image_path).convert("RGB")
		caption = row.get("caption", "")
		is_attacked = int(row.get("is_attacked", 0)) == 1
		return {"image": image, "caption": caption, "is_attacked": is_attacked}


def load_env() -> None:
	load_dotenv()
	if "HF_TOKEN" not in os.environ:
		print("Warning: HF_TOKEN is not set in the environment.")


def read_coco_metadata(coco_root: Path) -> list[dict[str, str]]:
	metadata_path = coco_root / "metadata.csv"
	if not metadata_path.exists():
		raise FileNotFoundError(f"Missing COCO metadata file: {metadata_path}")

	with metadata_path.open("r", encoding="utf-8", newline="") as handle:
		metadata = list(csv.DictReader(handle))

	required_columns = {"image_path", "main_caption"}
	missing_columns = required_columns.difference(metadata[0].keys() if metadata else set())
	if missing_columns:
		raise ValueError(f"COCO metadata is missing columns: {sorted(missing_columns)}")

	return metadata


def build_split_plan(
	metadata: list[dict[str, str]],
	train_size: int,
	test_size: int,
	attack_ratio: float,
	seed: int,
) -> list[SamplePlan]:
	total_size = train_size + test_size
	if len(metadata) < total_size:
		raise ValueError(
			f"Requested {total_size} samples, but only {len(metadata)} COCO images are available."
		)

	rng = random.Random(seed)
	source_indices = list(range(len(metadata)))
	rng.shuffle(source_indices)
	source_indices = source_indices[:total_size]

	train_indices = source_indices[:train_size]
	test_indices = source_indices[train_size:train_size + test_size]

	def make_attack_flags(size: int) -> list[bool]:
		attacked_count = int(round(size * attack_ratio))
		attacked_count = max(0, min(size, attacked_count))
		flags = [True] * attacked_count + [False] * (size - attacked_count)
		rng.shuffle(flags)
		return flags

	plans: list[SamplePlan] = []

	for split_name, indices in (("train", train_indices), ("test", test_indices)):
		attack_flags = make_attack_flags(len(indices))
		for local_index, (source_index, is_attacked) in enumerate(zip(indices, attack_flags, strict=True)):
			source_row = metadata[source_index]
			sample_id = f"{split_name}_{local_index:06d}"
			plans.append(
				SamplePlan(
					sample_id=sample_id,
					split=split_name,
					is_attacked=is_attacked,
					source_index=source_index,
					source_image_name=str(source_row["image_path"]),
					caption=str(source_row["main_caption"]),
				)
			)

	return plans


def ensure_output_dirs(root: Path) -> None:
	(root / "images" / "train").mkdir(parents=True, exist_ok=True)
	(root / "images" / "test").mkdir(parents=True, exist_ok=True)
	(root / "targets" / "train").mkdir(parents=True, exist_ok=True)
	(root / "targets" / "test").mkdir(parents=True, exist_ok=True)


def save_rgb_image(image: Image.Image, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	image.save(path, quality=95)


def save_quickdraw_target(target: torch.Tensor, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	target_uint8 = torch.clamp(target * 255.0, 0, 255).to(torch.uint8).cpu().numpy()
	Image.fromarray(target_uint8, mode="L").save(path)


def build_record_row(
	root: Path,
	sample_id: str,
	split: str,
	is_attacked: bool,
	source_image_path: Path,
	image_path: Path,
	caption: str,
) -> dict[str, Any]:
	# Minimal metadata: path relative to dataset root, caption, and attacked flag (0/1)
	return {
		"image_path": str(image_path.relative_to(root)),
		"caption": caption,
		"is_attacked": int(is_attacked),
	}


def generate_dataset_for_model(
	model_key: str,
	coco_root: Path,
	quickdraw_root: Path,
	output_root: Path,
	seed: int,
	train_size: int,
	test_size: int,
	attack_ratio: float,
) -> Path:
	coco_metadata = read_coco_metadata(coco_root)
	plans = build_split_plan(coco_metadata, train_size, test_size, attack_ratio, seed)

	dataset_root = output_root / f"{model_key}-adv-recognition-ds"
	if dataset_root.exists():
		print(f"Dataset root already exists: {dataset_root}")

	ensure_output_dirs(dataset_root)

	# write a tiny config for traceability
	(dataset_root / "info.json").write_text(
		json.dumps({"model_key": model_key, "seed": seed, "train_size": train_size, "test_size": test_size, "attack_ratio": attack_ratio}, indent=2),
		encoding="utf-8",
	)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Loading model {model_key} on {device}")
	model = build_clip_model(model_key, device=str(device), load_on_init=False)
	model.load_model()

	quickdraw_sampler = QuickDrawSampler(quickdraw_root, seed=seed + 17)
	records: list[dict[str, Any]] = []

	for plan in tqdm(plans, desc=f"Building {model_key}"):
		source_row = coco_metadata[plan.source_index]
		source_image_path = coco_root / "images" / str(source_row["image_path"])
		if not source_image_path.exists():
			raise FileNotFoundError(f"Missing source image: {source_image_path}")

		output_image_path = dataset_root / "images" / plan.split / f"{plan.sample_id}.jpg"
		output_target_path = dataset_root / "targets" / plan.split / f"{plan.sample_id}.png"

		image = Image.open(source_image_path).convert("RGB")
		image = image.resize(TARGET_SIZE, resample=Image.Resampling.LANCZOS)
		print(image.size)
		
		if plan.is_attacked:
			print(f"Attacking sample {plan.sample_id} with caption: {plan.caption}")
			target_tensor, _, _ = quickdraw_sampler.sample()
			print("target_tensor shape:", target_tensor.shape)
			attacked_image, _, _, _ = model.ruin(image, plan.caption, target_tensor)
			save_rgb_image(attacked_image.convert("RGB"), output_image_path)
			# targets are not needed in metadata for the simplified format, but saved for trace
			save_quickdraw_target(target_tensor, output_target_path)
			records.append(build_record_row(dataset_root, plan.sample_id, plan.split, True, source_image_path, output_image_path, plan.caption))
		else:
			print(f"Saving clean sample {plan.sample_id} with caption: {plan.caption}")
			save_rgb_image(image, output_image_path)
			records.append(build_record_row(dataset_root, plan.sample_id, plan.split, False, source_image_path, output_image_path, plan.caption))

		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	model.offload_from_gpu()
	model.unload_model()

	# only keep minimal columns in metadata: image_path, caption, is_attacked
	metadata_path = dataset_root / "metadata.csv"
	with metadata_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=["image_path", "caption", "is_attacked"]) 
		writer.writeheader()
		writer.writerows(records)

	# print(f"Finished {model_key}: {dataset_root}")
	# split_counts: dict[tuple[str, bool], int] = {}
	# for row in records:
	# 	key = (row["split"], bool(row["is_attacked"]))
	# 	split_counts[key] = split_counts.get(key, 0) + 1
	# print(split_counts)
	return dataset_root


def main() -> None:
	load_env()
	for model_key in SELECT_MODEL_KEYS:
		generate_dataset_for_model(
			model_key=model_key,
			coco_root=COCO_ROOT,
			quickdraw_root=QUICKDRAW_ROOT,
			output_root=OUTPUT_ROOT,
			seed=SEED,
			train_size=TRAIN_SIZE,
			test_size=TEST_SIZE,
			attack_ratio=ATTACK_RATIO,
		)


if __name__ == "__main__":
	main()
