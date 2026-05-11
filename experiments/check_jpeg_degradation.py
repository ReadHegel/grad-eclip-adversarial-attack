"""Check how much JPEG compression degrades the adversarial attack.

Compares attacked_heatmap_1head.npy (computed before JPEG save) with the
explanation recomputed from the saved perturbed.jpg. Saves a visualization
for each sample to results/jpeg_degradation_check/.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from models import build_clip_model
from experiments.multihead_explain import overlay_heatmap

RESULTS_DIR = Path("results/single_head_attack_test")
DATASET_DIR = Path("dataset")
OUT_DIR = Path("results/jpeg_degradation_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

results = json.load(open(RESULTS_DIR / "results.json"))
index = {r["id"]: r for r in json.load(open(DATASET_DIR / "index.json"))}

model = build_clip_model("openai-vit-b16")
model._set_num_heads(1)

print(f"{'id':15s}  {'mse_before_jpeg':>16s}  {'mse_after_jpeg':>14s}  {'ratio':>6s}")
print("-" * 60)

for r in results:
    sid = r["id"]
    sample_dir = RESULTS_DIR / sid
    record = index[sid]

    original_img = Image.open(DATASET_DIR / record["image_path"]).convert("RGB")
    target_np = np.load(DATASET_DIR / record["target_path"]).astype("float32")
    attacked_1h_before = np.load(sample_dir / "attacked_heatmap_1head.npy")
    clean_1h = np.load(sample_dir / "clean_heatmap_1head.npy")

    perturbed_img = Image.open(sample_dir / "perturbed.jpg").convert("RGB")
    emap, _ = model.explain(perturbed_img, record["caption"])
    attacked_1h_after = emap.detach().cpu().float().numpy()

    target_r = cv2.resize(target_np, (attacked_1h_before.shape[1], attacked_1h_before.shape[0]))

    mse_before = float(((attacked_1h_before - target_r) ** 2).mean())
    mse_after = float(((attacked_1h_after - target_r) ** 2).mean())
    ratio = mse_after / (mse_before + 1e-8)
    print(f"{sid}  {mse_before:>16.5f}  {mse_after:>14.5f}  {ratio:>6.2f}x")

    image_np = np.asarray(model._proccess_keepsize(original_img))
    attacked_np = np.asarray(model._proccess_keepsize(perturbed_img))

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(image_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(attacked_np); axes[1].set_title("Perturbed (from JPEG)"); axes[1].axis("off")
    axes[2].imshow(target_np, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Target"); axes[2].axis("off")
    axes[3].imshow(overlay_heatmap(torch.from_numpy(clean_1h), image_np))
    axes[3].set_title("Clean explain (1-head)"); axes[3].axis("off")
    axes[4].imshow(overlay_heatmap(torch.from_numpy(attacked_1h_before), image_np))
    axes[4].set_title(f"Attacked before JPEG\nMSE={mse_before:.4f}"); axes[4].axis("off")
    axes[5].imshow(overlay_heatmap(torch.from_numpy(attacked_1h_after), image_np))
    axes[5].set_title(f"Attacked after JPEG\nMSE={mse_after:.4f}  ratio={ratio:.2f}x"); axes[5].axis("off")

    plt.suptitle(record["caption"], fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{sid}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved viz → {OUT_DIR}/{sid}.png")

model.offload_from_gpu()
