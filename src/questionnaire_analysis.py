import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from src.config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT

# All self-reported columns present in the dataset
_QUEST_COLS = [
    "Frustrated",                                                   # 0-10 VAS
    "upset", "hostile", "ashamed", "nervous", "afraid",            # I-PANAS-SF negative
    "alert", "inspired", "attentive", "active", "determined",      # I-PANAS-SF positive
]


def run_questionnaire_correlation(
    Z_pca: np.ndarray,
    metadata: pd.DataFrame,
    n_pcs: int = 6,
) -> None:
    available = [c for c in _QUEST_COLS if c in metadata.columns]
    if not available:
        print("  No questionnaire columns in metadata — skipping.")
        return

    n_pcs = min(n_pcs, Z_pca.shape[1])
    pc_labels = [f"PC{i+1}" for i in range(n_pcs)]

    corr_mat = np.full((len(available), n_pcs), np.nan)
    pval_mat = np.full((len(available), n_pcs), np.nan)

    for i, col in enumerate(available):
        vals = pd.to_numeric(metadata[col], errors="coerce").values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            continue
        for j in range(n_pcs):
            r, p = pearsonr(Z_pca[mask, j], vals[mask])
            corr_mat[i, j] = r
            pval_mat[i, j] = p

    _plot_heatmap(corr_mat, pval_mat, available, pc_labels)
    _print_significant(corr_mat, pval_mat, available, pc_labels)


def _plot_heatmap(
    corr_mat: np.ndarray,
    pval_mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8, len(col_labels) * 1.3), max(5, len(row_labels) * 0.65))
    )

    mask = np.isnan(corr_mat)
    annot = np.where(mask, "", np.vectorize(lambda v: f"{v:.2f}")(corr_mat))

    sns.heatmap(
        corr_mat,
        annot=annot, fmt="",
        xticklabels=col_labels, yticklabels=row_labels,
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        mask=mask, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )

    # Mark significant cells with *
    for i in range(corr_mat.shape[0]):
        for j in range(corr_mat.shape[1]):
            if not np.isnan(pval_mat[i, j]) and pval_mat[i, j] < 0.05:
                ax.text(j + 0.5, i + 0.82, "*", ha="center", va="center",
                        color="black", fontsize=14, fontweight="bold")

    ax.set_title("PC Scores × Self-Reported Emotion (Pearson r)\n* = p < 0.05")
    ax.set_xlabel("Principal Component")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    path = FIGURES_DIR / "questionnaire_pc_correlation.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")


def _print_significant(
    corr_mat: np.ndarray,
    pval_mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    alpha: float = 0.05,
) -> None:
    print("\n  Significant PC–questionnaire correlations (p < 0.05):")
    found = False
    for i, qcol in enumerate(row_labels):
        for j, pclabel in enumerate(col_labels):
            if not np.isnan(pval_mat[i, j]) and pval_mat[i, j] < alpha:
                print(
                    f"    {pclabel} × {qcol}: "
                    f"r = {corr_mat[i, j]:+.3f},  p = {pval_mat[i, j]:.4f}"
                )
                found = True
    if not found:
        print("    None found.")
