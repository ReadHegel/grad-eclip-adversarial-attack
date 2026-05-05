"""Per-head Grad-ECLIP visualization for all registered CLIP models.

Computes Grad-ECLIP separately for each attention head of the last
transformer layer, plus the standard single-head baseline.

Per-head approach: slice Q, K, V and the gradient of cosine similarity
w.r.t. att_output along the embed_dim dimension.
Head h occupies positions [h*head_dim : (h+1)*head_dim].
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

from models import build_clip_model, CLIP_MODEL_REGISTRY

# ── config ────────────────────────────────────────────────────────────────────
IMAGE_PATH = "Images/SampleImages/dog_and_car.png"
TEXT = "a dog in a car waiting for traffic lights"
OUT_DIR = "Images/Output/multihead"


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
    q_raw: torch.Tensor,      # (seq_len, 1, embed_dim)
    k_raw: torch.Tensor,      # (seq_len, 1, embed_dim)
    v: torch.Tensor,          # (seq_len, 1, embed_dim)
    patch_map_size: tuple,
    head_idx: int,
    head_dim: int,
    cls_token: bool = True,
) -> torch.Tensor:
    """Grad-ECLIP heatmap for a single attention head."""
    s = head_idx * head_dim
    e = s + head_dim

    grad_cls_h = grad[:1, 0, s:e]
    q_cls_h    = q_raw[:1, 0, s:e]
    k_patch_h  = k_raw[:, 0, s:e]
    v_h        = v[:, 0, s:e]

    q_n = F.normalize(q_cls_h, dim=-1)
    k_n = F.normalize(k_patch_h, dim=-1)
    cosine_qk = (q_n * k_n).sum(-1)
    cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min() + 1e-8)

    emap = F.relu((grad_cls_h * v_h * cosine_qk[:, None]).sum(-1))

    if cls_token:
        emap = emap[1:]
    emap = emap.reshape(*patch_map_size)
    emap = (emap - emap.min()) / (emap.max() + 1e-8)
    return emap


def run_model(model_key: str, image: Image.Image, device: str) -> None:
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"{'='*60}")

    model = build_clip_model(model_key, device=device, load_on_init=False)
    model.load_model()

    # Read architecture params from model config
    vcfg = model.model.config.vision_config
    num_heads = vcfg.num_attention_heads
    embed_dim = vcfg.hidden_size
    head_dim  = embed_dim // num_heads
    print(f"  embed_dim={embed_dim}  num_heads={num_heads}  head_dim={head_dim}")

    # Preprocessed PIL image for overlays (resized to patch-grid multiples)
    image_proc = model._proccess_keepsize(image)
    image_np   = np.asarray(image_proc)

    # Pixel tensor
    pixel_values = model.proccess_keepsize(image).unsqueeze(0).to(device).detach()

    # Text embedding
    text_inputs = model.processor(text=[TEXT], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.model.get_text_features(**text_inputs)
    # HF CLIP returns a tensor; SigLIP returns BaseModelOutput — handle both
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    text_embedding = F.normalize(text_features, dim=-1)

    # Forward
    dense = model.encode_dense(pixel_values)
    img_embedding = F.normalize(dense.embeddings, dim=-1)
    c = (img_embedding @ text_embedding.T)[0][0]
    print(f"  cosine similarity: {c.item():.4f}")

    # Baseline single-head Grad-ECLIP
    emap_baseline = model._grad_eclip(
        c, dense.q_out, dense.k_out, dense.v,
        dense.att_output, dense.patch_map_size,
        withksim=True, cls_token=model.cls_token,
    )
    emap_baseline = (emap_baseline - emap_baseline.min()) / (emap_baseline.max() + 1e-8)

    # Compute gradient once (graph still alive after _grad_eclip with retain_graph=True)
    grad = torch.autograd.grad(c, dense.att_output, retain_graph=False)[0]

    # Per-head heatmaps
    emaps_per_head = []
    for h in range(num_heads):
        emap_h = head_grad_eclip(
            grad, dense.q_raw, dense.k_raw, dense.v,
            dense.patch_map_size,
            head_idx=h, head_dim=head_dim,
            cls_token=model.cls_token,
        )
        emaps_per_head.append(emap_h)
        print(f"  head {h:2d}: max={emap_h.max().item():.4f}  mean={emap_h.mean().item():.4f}")

    # Build panel list
    panels: list[tuple[str, np.ndarray]] = []
    panels.append(("Original", image_np))
    panels.append(("Baseline (1-head)", overlay_heatmap(emap_baseline, image_np)))
    for h, emap_h in enumerate(emaps_per_head):
        panels.append((f"Head {h}", overlay_heatmap(emap_h, image_np)))

    # Layout: 4 columns
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for i, (title, img) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for i in range(len(panels), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    plt.suptitle(f"{model_key}  |  \"{TEXT}\"", fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"multihead_{model_key}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")

    model.offload_from_gpu()
    model.unload_model()


def main() -> None:
    load_dotenv()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    os.makedirs(OUT_DIR, exist_ok=True)

    for model_key in CLIP_MODEL_REGISTRY:
        run_model(model_key, image, device)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
