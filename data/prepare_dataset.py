"""
Prepares the dataset for adversarial explanation manipulation evaluation.

Downloads 1000 images with captions from COCO val2017 and generates
random target masks for use as attack targets.

Usage:
    uv run python prepare_dataset.py --n 1000 --output dataset/ --seed 42
    uv run python prepare_dataset.py --n 5 --output dataset_test/ --seed 42  # quick test
"""

import argparse
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGES_BASE_URL = "http://images.cocodataset.org/val2017"
TARGET_SIZE = 64


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc, leave=False
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest


def ensure_annotations(cache_dir: Path) -> Path:
    captions_path = cache_dir / "annotations" / "captions_val2017.json"
    if captions_path.exists():
        return captions_path

    zip_path = cache_dir / "annotations_trainval2017.zip"
    print("Downloading COCO annotations (~240 MB)...")
    download_file(COCO_ANNOTATIONS_URL, zip_path, desc="annotations zip")

    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    zip_path.unlink()
    return captions_path


def load_captions(captions_path: Path) -> dict[int, dict]:
    """Returns {image_id: {filename, caption}}."""
    with open(captions_path) as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # keep only the first caption per image
    seen = set()
    result = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in seen:
            seen.add(iid)
            result[iid] = {
                "filename": id_to_filename[iid],
                "caption": ann["caption"].strip(),
            }
    return result


def download_image(filename: str, dest: Path) -> bool:
    if dest.exists():
        return True
    url = f"{COCO_IMAGES_BASE_URL}/{filename}"
    try:
        download_file(url, dest)
        return True
    except Exception as e:
        print(f"  Failed to download {filename}: {e}")
        return False


def make_blobs_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    n_blobs = rng.integers(1, 6)
    for _ in range(n_blobs):
        cx = rng.uniform(0, size)
        cy = rng.uniform(0, size)
        sigma = rng.uniform(3, 15)
        amplitude = rng.uniform(0.5, 1.0)
        xs, ys = np.meshgrid(np.arange(size), np.arange(size))
        blob = amplitude * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        mask += blob.astype(np.float32)
    return np.clip(mask, 0, 1)


def make_shape_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    shape = rng.choice(["circle", "rectangle"])
    if shape == "circle":
        cx = rng.uniform(size * 0.2, size * 0.8)
        cy = rng.uniform(size * 0.2, size * 0.8)
        r = rng.uniform(size * 0.1, size * 0.4)
        xs, ys = np.meshgrid(np.arange(size), np.arange(size))
        mask[(xs - cx) ** 2 + (ys - cy) ** 2 <= r**2] = 1.0
    else:
        x0 = int(rng.uniform(0, size * 0.5))
        y0 = int(rng.uniform(0, size * 0.5))
        x1 = int(rng.uniform(size * 0.5, size))
        y1 = int(rng.uniform(size * 0.5, size))
        mask[y0:y1, x0:x1] = 1.0
    mask = gaussian_filter(mask, sigma=1.0)
    return mask.astype(np.float32)


def make_noise_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    noise = rng.uniform(0, 1, (size, size)).astype(np.float32)
    sigma = rng.uniform(3, 8)
    blurred = gaussian_filter(noise, sigma=sigma)
    lo, hi = blurred.min(), blurred.max()
    if hi - lo > 1e-6:
        blurred = (blurred - lo) / (hi - lo)
    return blurred.astype(np.float32)


MASK_GENERATORS = {
    "blobs": make_blobs_mask,
    "shape": make_shape_mask,
    "noise": make_noise_mask,
}


def generate_target_mask(rng: np.random.Generator, size: int) -> tuple[np.ndarray, str]:
    mask_type = str(rng.choice(list(MASK_GENERATORS.keys())))
    mask = MASK_GENERATORS[mask_type](rng, size)
    return mask, mask_type


def save_mask(mask: np.ndarray, npy_path: Path, png_path: Path) -> None:
    np.save(npy_path, mask)
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img.save(png_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO-based adversarial attack dataset")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="dataset/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=str, default=".coco_cache/", help="Cache dir for COCO annotations")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    targets_dir = output_dir / "targets"
    images_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # --- Step 1: annotations ---
    captions_path = ensure_annotations(cache_dir)
    print("Loading captions...")
    all_samples = load_captions(captions_path)

    # --- Step 2: sample N image ids ---
    all_ids = list(all_samples.keys())
    if args.n > len(all_ids):
        raise ValueError(f"Requested {args.n} samples but COCO val has only {len(all_ids)}")
    selected_ids = rng.choice(all_ids, size=args.n, replace=False).tolist()

    # --- Step 3: download images ---
    print(f"Downloading {args.n} images...")
    download_tasks = [
        (all_samples[iid]["filename"], images_dir / all_samples[iid]["filename"])
        for iid in selected_ids
    ]
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, fname, dest): iid for iid, (fname, dest) in zip(selected_ids, download_tasks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="images"):
            iid = futures[future]
            if not future.result():
                failed.append(iid)

    if failed:
        print(f"Warning: {len(failed)} images failed to download, removing from index")
        selected_ids = [iid for iid in selected_ids if iid not in set(failed)]

    # --- Step 4: generate target masks ---
    print("Generating target masks...")
    index = []
    for iid in tqdm(selected_ids, desc="masks"):
        sample = all_samples[iid]
        sample_id = f"{iid:012d}"
        mask, mask_type = generate_target_mask(rng, TARGET_SIZE)
        npy_path = targets_dir / f"target_{sample_id}.npy"
        png_path = targets_dir / f"target_{sample_id}.png"
        save_mask(mask, npy_path, png_path)

        index.append({
            "id": sample_id,
            "coco_image_id": iid,
            "image_path": str(Path("images") / sample["filename"]),
            "caption": sample["caption"],
            "target_path": str(Path("targets") / f"target_{sample_id}.npy"),
            "target_type": mask_type,
        })

    # --- Step 5: save index + splits ---
    rng.shuffle(index)

    n = len(index)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": index[:n_train],
        "val": index[n_train : n_train + n_val],
        "test": index[n_train + n_val :],
    }

    for split_name, records in splits.items():
        for rec in records:
            rec["split"] = split_name

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    for split_name, records in splits.items():
        split_path = output_dir / f"{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(records, f, indent=2)

    type_counts = {}
    for rec in index:
        type_counts[rec["target_type"]] = type_counts.get(rec["target_type"], 0) + 1

    print(f"\nDone. {len(index)} samples saved to {output_dir}/")
    print(f"  images:  {images_dir}/")
    print(f"  targets: {targets_dir}/  {type_counts}")
    print(f"  index:   {index_path}")
    print(f"  splits:  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


if __name__ == "__main__":
    main()
