"""PCC-based analysis: per-head correlation with the clean 1-head explanation.

Setup: model is configured with 1 attention head (original Grad-ECLIP inference mode).
The attack perturbs the image to move the 1-head explanation toward a target mask.
After the attack, the model is switched to 12 heads to analyse which heads still
produce explanations correlated with the original clean 1-head explanation.

For each sample three metrics are computed (all vs clean_1h):
  A. PCC(clean_12h[h],    clean_1h)  -- baseline: which heads match 1-head on clean image
  B. PCC(attacked_1h,     clean_1h)  -- how much did the 1-head attack shift the explanation
  C. PCC(attacked_12h[h], clean_1h)  -- which heads on attacked image still match original

Output: one PNG per sample + one overview figure.

Usage:
    uv run python -m experiments.analyze_pcc_vs_clean \\
        --results results/single_head_attack_test \\
        --output  results/pcc_vs_clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    r, _ = pearsonr(a.ravel().astype(np.float64), b.ravel().astype(np.float64))
    return float(r)


def plot_sample(
    sid: str,
    A: np.ndarray,  # (n_heads,)
    B: float,
    C: np.ndarray,  # (n_heads,)
    caption: str,
    out_path: Path,
) -> None:
    n_heads = len(A)
    xs = np.arange(n_heads)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(xs, A, "o-", color="steelblue", label="A: clean 12h vs clean 1h", lw=1.5)
    ax.plot(xs, C, "s-", color="tomato",    label="C: attacked 12h vs clean 1h", lw=1.5)
    ax.axhline(B, ls="--", color="tomato", alpha=0.5, lw=1,
               label=f"B: attacked 1h vs clean 1h = {B:.3f}")
    ax.axhline(0, ls=":", color="gray", lw=0.8)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("PCC vs clean_1h")
    ax.set_ylim(-1, 1)
    ax.set_title(f'{sid}\n"{caption}"', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/single_head_attack_test")
    parser.add_argument("--output", default="results/pcc_vs_clean")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = json.load(open(results_dir / "results.json"))
    n_heads = None

    all_A: list[np.ndarray] = []
    all_B: list[float] = []
    all_C: list[np.ndarray] = []

    for r in records:
        sid = r["id"]
        d = results_dir / sid

        clean_1h    = np.load(d / "clean_heatmap_1head.npy").astype(np.float32)
        attacked_1h = np.load(d / "attacked_heatmap_1head.npy").astype(np.float32)
        clean_12h   = np.load(d / "clean_heatmaps_12head.npy").astype(np.float32)
        attacked_12h = np.load(d / "attacked_heatmaps_12head.npy").astype(np.float32)

        if n_heads is None:
            n_heads = clean_12h.shape[0]

        A = np.array([pcc(clean_12h[h], clean_1h)    for h in range(n_heads)])
        B = pcc(attacked_1h, clean_1h)
        C = np.array([pcc(attacked_12h[h], clean_1h) for h in range(n_heads)])

        all_A.append(A)
        all_B.append(B)
        all_C.append(C)

        plot_sample(sid, A, B, C, r["caption"], out_dir / f"{sid}.png")
        print(f"saved {out_dir}/{sid}.png  (B={B:.3f})")

    # ── overview: all samples in one figure ─────────────────────────────────
    n = len(records)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), sharey=True)
    axes_flat = axes.ravel()
    xs = np.arange(n_heads)

    for i, (r, A, B, C) in enumerate(zip(records, all_A, all_B, all_C)):
        ax = axes_flat[i]
        ax.plot(xs, A, "o-", color="steelblue", lw=1.2, ms=4)
        ax.plot(xs, C, "s-", color="tomato",    lw=1.2, ms=4)
        ax.axhline(B, ls="--", color="tomato", alpha=0.5, lw=1)
        ax.axhline(0, ls=":", color="gray", lw=0.7)
        ax.set_title(r["id"], fontsize=7)
        ax.set_xticks(xs[::2])
        ax.set_xticklabels([f"H{h}" for h in range(0, n_heads, 2)], fontsize=6)
        ax.set_ylim(-1, 1)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    # legend in first panel
    axes_flat[0].plot([], [], "o-", color="steelblue", label="A: clean 12h vs clean 1h")
    axes_flat[0].plot([], [], "s-", color="tomato",    label="C: attacked 12h vs clean 1h")
    axes_flat[0].plot([], [], "--", color="tomato", alpha=0.5, label="B: attacked 1h vs clean 1h")
    axes_flat[0].legend(fontsize=6, loc="lower right")

    fig.suptitle("PCC vs clean_1h — per sample (blue=clean 12h, red=attacked 12h, dashed=attacked 1h)",
                 fontsize=9)
    fig.tight_layout()
    path = out_dir / "overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved overview → {path}")


if __name__ == "__main__":
    main()
