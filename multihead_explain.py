"""Per-head Grad-ECLIP visualization.

Computes Grad-ECLIP separately for each of the 12 attention heads
of openai-vit-b16 and saves a comparison PNG.

Per-head approach: slice Q, K, V and the gradient of cosine similarity
w.r.t. att_output along the embed_dim dimension (head h occupies
positions [h*head_dim : (h+1)*head_dim]).
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from dotenv import load_dotenv
from PIL import Image

from models import build_clip_model

# ── config ────────────────────────────────────────────────────────────────────
IMAGE_PATH = "Images/SampleImages/dog_and_car.png"
TEXT = "a dog in a car waiting for traffic lights"
MODEL_KEY = "openai-vit-b16"
OUT_PATH = "Images/Output/multihead_explain.png"

NUM_HEADS = 12
EMBED_DIM = 768
HEAD_DIM = EMBED_DIM // NUM_HEADS  # 64


# ── helpers ───────────────────────────────────────────────────────────────────

def overlay_heatmap(hmap: torch.Tensor, image_np: np.ndarray) -> np.ndarray:
    """Overlay a (H, W) heatmap tensor on an RGB numpy image."""
    h, w = image_np.shape[:2]
    resize = torchvision.transforms.Resize(
        (h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST
    )
    hmap_np = resize(hmap.unsqueeze(0))[0].detach().cpu().numpy()
    color = cv2.applyColorMap((hmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.clip(image_np * 0.5 + color * 0.5, 0, 255).astype(np.uint8)


def head_grad_eclip(
    grad: torch.Tensor,       # (seq_len, 1, embed_dim) — pre-computed gradient
    q_raw: torch.Tensor,      # (seq_len, 1, embed_dim) — raw Q from q_proj
    k_raw: torch.Tensor,      # (seq_len, 1, embed_dim) — raw K from k_proj
    v: torch.Tensor,          # (seq_len, 1, embed_dim) — raw V from v_proj
    patch_map_size: tuple,
    head_idx: int,
    cls_token: bool = True,
) -> torch.Tensor:
    """Grad-ECLIP heatmap for a single attention head."""
    s = head_idx * HEAD_DIM
    e = s + HEAD_DIM

    grad_cls_h = grad[:1, 0, s:e]       # (1, head_dim)
    q_cls_h = q_raw[:1, 0, s:e]         # (1, head_dim)
    k_patch_h = k_raw[:, 0, s:e]        # (seq_len, head_dim)
    v_h = v[:, 0, s:e]                  # (seq_len, head_dim)

    q_n = F.normalize(q_cls_h, dim=-1)
    k_n = F.normalize(k_patch_h, dim=-1)
    cosine_qk = (q_n * k_n).sum(-1)     # (seq_len,)
    cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min() + 1e-8)

    emap = F.relu((grad_cls_h * v_h * cosine_qk[:, None]).sum(-1))  # (seq_len,)

    if cls_token:
        emap = emap[1:]  # drop CLS position
    emap = emap.reshape(*patch_map_size)
    emap = (emap - emap.min()) / (emap.max() + 1e-8)
    return emap


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image = Image.open(IMAGE_PATH).convert("RGB")

    model = build_clip_model(MODEL_KEY, device=device, load_on_init=False)
    model.load_model()

    # Preprocessed PIL image (resized to patch-grid multiples) for overlays
    image_proc = model._proccess_keepsize(image)
    image_np = np.asarray(image_proc)

    # Pixel tensor for encode_dense
    pixel_values = model.proccess_keepsize(image).unsqueeze(0).to(device).detach()

    # Text embedding (detached — we only need grad w.r.t. attention internals)
    text_inputs = model.processor(text=[TEXT], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.model.get_text_features(**text_inputs).pooler_output
    text_embedding = F.normalize(text_features, dim=-1)

    # ── forward pass ──────────────────────────────────────────────────────────
    dense = model.encode_dense(pixel_values)
    img_embedding = F.normalize(dense.embeddings, dim=-1)
    c = (img_embedding @ text_embedding.T)[0][0]
    print(f"Cosine similarity: {c.item():.4f}")

    # ── baseline: single-head Grad-ECLIP ─────────────────────────────────────
    emap_baseline = model._grad_eclip(
        c, dense.q_out, dense.k_out, dense.v,
        dense.att_output, dense.patch_map_size,
        withksim=True, cls_token=model.cls_token,
    )
    emap_baseline = (emap_baseline - emap_baseline.min()) / (emap_baseline.max() + 1e-8)

    # ── per-head: compute gradient once, then slice per head ─────────────────
    # _grad_eclip already used retain_graph=True so the graph is still alive.
    grad = torch.autograd.grad(c, dense.att_output, retain_graph=False)[0]
    # grad shape: (seq_len, 1, embed_dim)

    emaps_per_head = []
    for h in range(NUM_HEADS):
        emap_h = head_grad_eclip(
            grad, dense.q_raw, dense.k_raw, dense.v,
            dense.patch_map_size, head_idx=h, cls_token=model.cls_token,
        )
        emaps_per_head.append(emap_h)
        print(f"Head {h:2d}: max={emap_h.max().item():.4f}  mean={emap_h.mean().item():.4f}")

    # ── visualize ─────────────────────────────────────────────────────────────
    # Layout: 4 columns
    # Row 0: original image | baseline | head 0 | head 1
    # Row 1: head 2  … head 5
    # Row 2: head 6  … head 9
    # Row 3: head 10 | head 11 | (empty) | (empty)

    panels: list[tuple[str, np.ndarray]] = []
    panels.append(("Original", image_np))
    panels.append(("Baseline (1-head)", overlay_heatmap(emap_baseline, image_np)))
    for h, emap_h in enumerate(emaps_per_head):
        panels.append((f"Head {h}", overlay_heatmap(emap_h, image_np)))

    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    for i, (title, img) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for i in range(len(panels), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    plt.suptitle(
        f"{MODEL_KEY}  |  \"{TEXT}\"",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
