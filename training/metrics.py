"""Binary classification metrics (ROC / AUC) without extra dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute FPR/TPR points and ROC-AUC for binary labels (positive class = 1).

    Returns (fpr, tpr, auc) with (0, 0) prepended to the curve.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    *,
    title: str,
) -> float:
    """Save ROC plot to path; return AUC (nan if undefined)."""
    import matplotlib.pyplot as plt

    fpr, tpr, auc = binary_roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}" if not np.isnan(auc) else "AUC = n/a")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return auc
