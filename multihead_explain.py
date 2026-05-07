"""Per-head Grad-ECLIP visualization for all registered CLIP models.

Output: Images/Output/multihead/multihead_{model_key}.png
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image

from models import build_clip_model, CLIP_MODEL_REGISTRY

IMAGE_PATH = "Images/Output/all_heads_attack/openai-vit-b16_perturbed_all_heads.png"
TEXT = "a dog in a car waiting for traffic lights"
OUT_DIR = "Images/Output/multihead/explain_whith12_attacked_on6"
NUM_HEADS = 12
MODEL_KEY = "openai-vit-b16"


def overlay_heatmap(hmap: torch.Tensor, image_np: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    resize = torchvision.transforms.Resize((h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    hmap_np = resize(hmap.unsqueeze(0))[0].detach().cpu().numpy()
    color = cv2.applyColorMap((hmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.clip(image_np * 0.5 + color * 0.5, 0, 255).astype(np.uint8)


def run_model(model_key: str, image: Image.Image) -> None:
    print(f"\n=== {model_key} ===")
    model = build_clip_model(model_key, num_heads=NUM_HEADS)

    vcfg = model.model.config.vision_config
    num_heads = vcfg.num_attention_heads
    print(f"  num_heads={num_heads}  embed_dim={vcfg.hidden_size}")

    image_np = np.asarray(model._proccess_keepsize(image))
    emap_baseline, _ = model.explain(image, TEXT)
    emaps_per_head = model.explain_per_head(image, TEXT)

    panels = [("Original", image_np), ("Baseline (1-head)", overlay_heatmap(emap_baseline, image_np))]
    panels += [(f"Head {h}", overlay_heatmap(e, image_np)) for h, e in enumerate(emaps_per_head)]

    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for i, (title, img) in enumerate(panels):
        axes[i // ncols][i % ncols].imshow(img)
        axes[i // ncols][i % ncols].set_title(title, fontsize=11)
        axes[i // ncols][i % ncols].axis("off")
    for i in range(len(panels), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    plt.suptitle(f'{model_key}  |  "{TEXT}"', fontsize=11, y=1.01)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, f"multihead_{model_key}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT_DIR}/multihead_{model_key}.png")

    model.offload_from_gpu()
    model.unload_model()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    image = Image.open(IMAGE_PATH).convert("RGB")
    # for model_key in CLIP_MODEL_REGISTRY:
    #     run_model(model_key, image)
    run_model(model_key=MODEL_KEY, image=image)
