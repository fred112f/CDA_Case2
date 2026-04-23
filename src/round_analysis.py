import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

from src.config import (
    FIGURES_DIR, PHASE_COL, ROUND_COL,
    PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)


def run_round_analysis(
    Z_pca: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    """
    Compute phase silhouette score separately for each round in PC1-2 space.
    Also plots faceted PCA projections so you can visually compare rounds.
    Answers: is phase structure consistent, or stronger in specific rounds?
    """
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if label_col not in metadata.columns:
        print("  No phase labels — skipping round analysis.")
        return
    if ROUND_COL not in metadata.columns:
        print("  No round column — skipping round analysis.")
        return

    rounds = sorted(metadata[ROUND_COL].dropna().unique())
    Z2 = Z_pca[:, :2]
    labels_all = metadata[label_col].values

    sil_results: list[tuple] = []
    for rnd in rounds:
        mask = (metadata[ROUND_COL] == rnd).values
        Z_rnd = Z2[mask]
        labels_rnd = labels_all[mask]
        n_cls = len(np.unique(labels_rnd))
        if n_cls < 2 or Z_rnd.shape[0] < 6:
            continue
        sil = silhouette_score(Z_rnd, labels_rnd)
        sil_results.append((rnd, sil, int(mask.sum())))
        print(f"  Round {rnd}: silhouette = {sil:.3f}  (n = {mask.sum()})")

    if sil_results:
        _plot_silhouette_by_round(sil_results)
    _plot_pca_by_round(Z2, metadata, label_col, rounds)


def _plot_silhouette_by_round(sil_results: list[tuple]) -> None:
    labels = [str(r) for r, _, _ in sil_results]
    values = [s for _, s, _ in sil_results]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4))
    colors = ["#C44E52" if s < 0 else "#4C72B0" for s in values]
    ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Silhouette Score (phase labels, PC1–2)")
    ax.set_title("Phase Separation per Round\n"
                 "Positive = phases cluster in PC space; negative = overlap")
    plt.tight_layout()

    path = FIGURES_DIR / "round_silhouette.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_pca_by_round(
    Z2: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str,
    rounds: list,
) -> None:
    n_rounds = len(rounds)
    n_cols = min(4, n_rounds)
    n_rows = (n_rounds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    existing = set(metadata[label_col].dropna().unique())
    pal = {k: v for k, v in PHASE_PALETTE.items() if k in existing}

    df_base = metadata[[ROUND_COL, label_col]].copy()
    df_base["PC1"] = Z2[:, 0]
    df_base["PC2"] = Z2[:, 1]

    for ax_idx, rnd in enumerate(rounds):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]
        mask = (metadata[ROUND_COL] == rnd).values
        sub = df_base[mask]

        sns.scatterplot(
            data=sub, x="PC1", y="PC2", hue=label_col,
            palette=pal or None, ax=ax, alpha=0.75, s=55, linewidth=0,
            legend=(ax_idx == 0),
        )
        ax.set_title(f"Round: {rnd}  (n = {mask.sum()})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")
        if ax_idx == 0:
            leg = ax.get_legend()
            if leg:
                leg.set_title("Phase")

    for ax_idx in range(len(rounds), n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("PCA Projections by Round (PC1 vs PC2)", fontsize=12)
    plt.tight_layout()

    path = FIGURES_DIR / "round_pca_facets.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
