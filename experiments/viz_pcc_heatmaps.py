"""Visualise per-head explanations of attacked images with PCC labels.

For each sample generates one PNG:
  - col 0: original image
  - col 1: clean 1-head explanation overlay  (reference, PCC=1.0 by definition)
  - cols 2-13: attacked_12h[h] overlays, each labelled with PCC vs clean_1h

Usage:
    uv run python -m experiments.viz_pcc_heatmaps \\
        --results results/single_head_attack_test \\
        --dataset dataset \\
        --output  results/pcc_heatmap_viz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import pearsonr


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    r, _ = pearsonr(a.ravel().astype(np.float64), b.ravel().astype(np.float64))
    return float(r)


def overlay(hmap: np.ndarray, image_np: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    hmap_r = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((hmap_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.clip(image_np * 0.5 + color * 0.5, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/single_head_attack_test")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output",  default="results/pcc_heatmap_viz")
    args = parser.parse_args()

    results_dir = Path(args.results)
    dataset_dir = Path(args.dataset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    records     = json.load(open(results_dir / "results.json"))
    index       = {r["id"]: r for r in json.load(open(dataset_dir / "index.json"))}

    for r in records:
        sid    = r["id"]
        rec    = index[sid]
        d      = results_dir / sid

        image_np     = np.asarray(Image.open(dataset_dir / rec["image_path"]).convert("RGB"))
        clean_1h     = np.load(d / "clean_heatmap_1head.npy").astype(np.float32)
        attacked_12h = np.load(d / "attacked_heatmaps_12head.npy").astype(np.float32)

        n_heads = attacked_12h.shape[0]
        pccs = [pcc(attacked_12h[h], clean_1h) for h in range(n_heads)]

        # layout: original | clean_1h | H0..H11  →  14 panels, 7 cols × 2 rows
        ncols = 7
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

        panels = (
            [("original", image_np), ("clean 1h\n(reference)", overlay(clean_1h, image_np))]
            + [(f"H{h}\nPCC={pccs[h]:.3f}", overlay(attacked_12h[h], image_np))
               for h in range(n_heads)]
        )

        for i, (title, img) in enumerate(panels):
            ax = axes[i // ncols][i % ncols]
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        # blank remaining panels
        for i in range(len(panels), nrows * ncols):
            axes[i // ncols][i % ncols].axis("off")

        fig.suptitle(f'{sid}  |  "{rec["caption"]}"', fontsize=8, y=1.01)
        fig.tight_layout()
        path = out_dir / f"{sid}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
