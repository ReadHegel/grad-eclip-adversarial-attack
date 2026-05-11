"""Experiment: attack explanation in 1-head mode, observe effect across all 12 heads.

For each dataset record:
  1. Configure model with 1 attention head
  2. Compute clean 1-head explanation
  3. Run ruin_head(attack_head=0) against target mask
  4. Switch to 12 heads, compute explain_per_head() on clean + attacked image
  5. Save numpy arrays + loss curve; optionally save visualization

Output layout:
  <output_dir>/
    results.json                          # per-sample summary
    <id>/
      perturbed.jpg
      clean_heatmap_1head.npy            # (H, W) float32
      attacked_heatmap_1head.npy         # (H, W) float32
      clean_heatmaps_12head.npy          # (12, H, W) float32
      attacked_heatmaps_12head.npy       # (12, H, W) float32
      losses.npy                         # (steps,) float32
      viz.png                            # optional, every --viz-every N samples

Usage:
    uv run python experiments/experiment_single_head_attack_dataset.py \\
        --dataset dataset/ --output results/single_head_attack \\
        --model openai-vit-b16 --split train --n 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from models import build_clip_model


def overlay_heatmap(hmap: torch.Tensor | np.ndarray, image_np: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    if isinstance(hmap, torch.Tensor):
        resize = torchvision.transforms.Resize((h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        hmap_np = resize(hmap.unsqueeze(0))[0].detach().cpu().float().numpy()
    else:
        hmap_np = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((hmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.clip(image_np * 0.5 + color * 0.5, 0, 255).astype(np.uint8)


def save_viz(
    sample_dir: Path,
    image_np: np.ndarray,
    attacked_np: np.ndarray,
    target_np: np.ndarray,
    clean_1h: np.ndarray,
    attacked_1h: np.ndarray,
    clean_12h: np.ndarray,
    attacked_12h: np.ndarray,
    losses: np.ndarray,
    record: dict,
) -> None:
    num_heads = clean_12h.shape[0]
    # rows: header row + num_heads rows
    nrows = 1 + num_heads
    ncols = 5  # orig, attacked, target, clean-12h, attacked-12h

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))

    # --- header row ---
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original", fontsize=9)

    axes[0, 1].imshow(attacked_np)
    axes[0, 1].set_title("Attacked", fontsize=9)

    axes[0, 2].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Target mask", fontsize=9)

    axes[0, 3].imshow(overlay_heatmap(clean_1h, image_np))
    axes[0, 3].set_title("Clean (1-head)", fontsize=9)

    axes[0, 4].imshow(overlay_heatmap(attacked_1h, image_np))
    axes[0, 4].set_title("Attacked (1-head)", fontsize=9)

    for ax in axes[0]:
        ax.axis("off")

    # loss inset on top-right
    ax_loss = axes[0, 4].inset_axes([0.0, -0.45, 1.0, 0.4])
    ax_loss.plot(losses, lw=1)
    ax_loss.set_title("Loss", fontsize=7)
    ax_loss.tick_params(labelsize=6)

    # --- per-head rows ---
    for h in range(num_heads):
        axes[h + 1, 0].axis("off")
        axes[h + 1, 1].axis("off")
        axes[h + 1, 2].axis("off")

        axes[h + 1, 3].imshow(overlay_heatmap(clean_12h[h], image_np))
        axes[h + 1, 3].set_title(f"Head {h} — clean", fontsize=8)
        axes[h + 1, 3].axis("off")

        axes[h + 1, 4].imshow(overlay_heatmap(attacked_12h[h], image_np))
        axes[h + 1, 4].set_title(f"Head {h} — attacked", fontsize=8)
        axes[h + 1, 4].axis("off")

    plt.suptitle(f'id={record["id"]}  |  "{record["caption"]}"', fontsize=9, y=1.005)
    plt.tight_layout()
    fig.savefig(sample_dir / "viz.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def process_sample(model, record: dict, dataset_dir: Path, sample_dir: Path, args) -> dict:
    image_path = dataset_dir / record["image_path"]
    target_path = dataset_dir / record["target_path"]

    image = Image.open(image_path).convert("RGB")
    target_np = np.load(target_path).astype(np.float32)
    target_tensor = torch.from_numpy(target_np)
    text = record["caption"]

    # --- 1-head attack ---
    model._set_num_heads(1)

    # baseline 1-head explanation
    clean_1h_tensor, _ = model.explain(image, text)
    clean_1h = clean_1h_tensor.detach().cpu().float().numpy()

    # attack
    attacked_img, losses = model.ruin_head(image, text, target_tensor, attack_head=0, DELTA=args.delta)

    # 1-head explanation of attacked image
    attacked_1h_tensor, _ = model.explain(attacked_img, text)
    attacked_1h = attacked_1h_tensor.detach().cpu().float().numpy()

    # --- 12-head explanations ---
    model._set_num_heads(args.num_heads)

    clean_12h_tensors = model.explain_per_head(image, text)
    attacked_12h_tensors = model.explain_per_head(attacked_img, text)

    clean_12h = np.stack([t.detach().cpu().float().numpy() for t in clean_12h_tensors])
    attacked_12h = np.stack([t.detach().cpu().float().numpy() for t in attacked_12h_tensors])
    losses_np = np.array(losses, dtype=np.float32)

    # --- save ---
    sample_dir.mkdir(parents=True, exist_ok=True)
    attacked_img.save(sample_dir / "perturbed.jpg")
    np.save(sample_dir / "clean_heatmap_1head.npy", clean_1h)
    np.save(sample_dir / "attacked_heatmap_1head.npy", attacked_1h)
    np.save(sample_dir / "clean_heatmaps_12head.npy", clean_12h)
    np.save(sample_dir / "attacked_heatmaps_12head.npy", attacked_12h)
    np.save(sample_dir / "losses.npy", losses_np)

    return {
        "id": record["id"],
        "split": record["split"],
        "caption": record["caption"],
        "target_type": record["target_type"],
        "final_loss": float(losses_np[-1]),
        "target_mse_1head": float(((attacked_1h - cv2.resize(target_np, (attacked_1h.shape[1], attacked_1h.shape[0]))) ** 2).mean()),
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/", help="Path to dataset root (contains index.json)")
    parser.add_argument("--output", default="results/single_head_attack", help="Output directory")
    parser.add_argument("--model", default="openai-vit-b16", help="Model key")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--n", type=int, default=None, help="Limit to N samples (for testing)")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of heads for post-attack analysis")
    parser.add_argument("--delta", type=float, default=0.03, help="Attack perturbation bound")
    parser.add_argument("--viz-every", type=int, default=10, help="Save viz for every N-th sample (0 = never)")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed samples")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load index
    if args.split == "all":
        records = json.load(open(dataset_dir / "index.json"))
    else:
        records = json.load(open(dataset_dir / f"{args.split}.json"))

    if args.n is not None:
        records = records[: args.n]

    print(f"Processing {len(records)} samples with model={args.model}, num_heads_analysis={args.num_heads}")

    model = build_clip_model(args.model)

    results_path = output_dir / "results.json"
    results: list[dict] = []
    if args.resume and results_path.exists():
        results = json.load(open(results_path))
    done_ids = {r["id"] for r in results}

    for i, record in enumerate(tqdm(records, desc="samples")):
        if record["id"] in done_ids:
            continue

        sample_dir = output_dir / record["id"]
        result = process_sample(model, record, dataset_dir, sample_dir, args)
        results.append(result)

        if args.viz_every > 0 and (i % args.viz_every == 0):
            image = Image.open(dataset_dir / record["image_path"]).convert("RGB")
            target_np = np.load(dataset_dir / record["target_path"]).astype(np.float32)
            attacked_img = Image.open(sample_dir / "perturbed.jpg")
            image_np = np.asarray(model._proccess_keepsize(image))
            attacked_np = np.asarray(model._proccess_keepsize(attacked_img))
            save_viz(
                sample_dir,
                image_np,
                attacked_np,
                target_np,
                np.load(sample_dir / "clean_heatmap_1head.npy"),
                np.load(sample_dir / "attacked_heatmap_1head.npy"),
                np.load(sample_dir / "clean_heatmaps_12head.npy"),
                np.load(sample_dir / "attacked_heatmaps_12head.npy"),
                np.load(sample_dir / "losses.npy"),
                record,
            )

        # checkpoint after every sample
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}")
    print(f"Total samples: {len(results)}, mean final loss: {np.mean([r['final_loss'] for r in results]):.4f}")

    model.offload_from_gpu()
    model.unload_model()


if __name__ == "__main__":
    main()
