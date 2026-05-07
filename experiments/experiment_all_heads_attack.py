"""Experiment: attack all heads simultaneously with summed loss.

Loads a smiley target, runs ruin_all_heads() where loss = sum of MSE across
all 12 heads, then measures the effect on each head with explain_per_head().

Output: Images/Output/all_heads_attack/{model_key}_attack_all_heads.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image

from models import build_clip_model
from experiments.multihead_explain import overlay_heatmap

IMAGE_PATH = "Images/SampleImages/dog_and_car.png"
TARGET_PATH = "Images/targets/smiley.gif"
TEXT = "a dog in a car waiting for traffic lights"
MODEL_KEY = "openai-vit-b16"
OUT_DIR = "Images/Output/all_heads_attack/1head"
NUM_HEADS = 1


def make_figure(image_np, target_np, emaps_before, emaps_after, model_key):
    num_heads = len(emaps_before)
    mse = [(emaps_after[h] - emaps_before[h]).pow(2).mean().item() for h in range(num_heads)]

    fig, axes = plt.subplots(num_heads + 1, 3, figsize=(12, 3 * (num_heads + 1)))

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_np, cmap="gray")
    axes[0, 1].set_title("Target (all heads)", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].bar(range(num_heads), mse, color="steelblue")
    axes[0, 2].set(xlabel="Head", ylabel="MSE before→after", title="Per-head impact (all attacked)")
    axes[0, 2].set_xticks(range(num_heads))

    for h in range(num_heads):
        axes[h + 1, 0].imshow(overlay_heatmap(emaps_before[h], image_np))
        axes[h + 1, 0].set_title(f"Head {h} — before", fontsize=8)
        axes[h + 1, 0].axis("off")

        axes[h + 1, 1].imshow(overlay_heatmap(emaps_after[h], image_np))
        axes[h + 1, 1].set_title(f"Head {h} — after", fontsize=8)
        axes[h + 1, 1].axis("off")

        diff = (emaps_after[h] - emaps_before[h]).abs()
        diff = (diff - diff.min()) / (diff.max() + 1e-8)
        resize = torchvision.transforms.Resize(
            image_np.shape[:2],
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        diff_np = resize(diff.unsqueeze(0))[0].detach().cpu().numpy()
        axes[h + 1, 2].imshow(diff_np, cmap="hot", vmin=0, vmax=1)
        axes[h + 1, 2].set_title(f"Head {h} — |Δ| mse={mse[h]:.4f}", fontsize=8)
        axes[h + 1, 2].axis("off")

    plt.suptitle(f'{model_key}  |  "{TEXT}"  |  all heads attacked', fontsize=10, y=1.01)
    plt.tight_layout()
    return fig


def main() -> None:
    load_dotenv()

    image = Image.open(IMAGE_PATH).convert("RGB")
    target_tensor = 1 - torchvision.transforms.ToTensor()(Image.open(TARGET_PATH).convert("RGBA"))[0, :, :]

    model = build_clip_model(MODEL_KEY, num_heads=NUM_HEADS)

    image_np = np.asarray(model._proccess_keepsize(image))
    target_np = target_tensor.numpy()

    print("Computing baseline explanations...")
    emaps_before = model.explain_per_head(image, TEXT)

    print("Attacking all heads simultaneously...")
    ruined_img, losses = model.ruin_all_heads(image, TEXT, target_tensor)

    print("Computing post-attack explanations...")
    emaps_after = model.explain_per_head(ruined_img, TEXT)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = make_figure(image_np, target_np, emaps_before, emaps_after, MODEL_KEY)
    out_path = os.path.join(OUT_DIR, f"{MODEL_KEY}_attack_all_heads.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    ruined_img.save(os.path.join(OUT_DIR, f"{MODEL_KEY}_perturbed_all_heads.png"))

    model.offload_from_gpu()
    model.unload_model()


if __name__ == "__main__":
    main()
