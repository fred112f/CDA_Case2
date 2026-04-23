import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap

from src.config import (
    FIGURES_DIR,
    TSNE_PERPLEXITY, TSNE_MAX_ITER, TSNE_RANDOM_STATE,
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_RANDOM_STATE,
    PHASE_COL, COHORT_COL, SUBJECT_COL, PUZZLER_COL,
    PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)


def run_tsne(X_scaled: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_MAX_ITER,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    Z = tsne.fit_transform(X_scaled)
    _plot_embedding(
        Z, metadata, method="t-SNE", prefix="tsne",
        subtitle=f"perplexity={TSNE_PERPLEXITY}",
    )
    _print_silhouette(Z, metadata, "t-SNE")
    print(f"  t-SNE KL divergence: {tsne.kl_divergence_:.3f}")
    return Z


def run_umap(X_scaled: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=UMAP_RANDOM_STATE,
    )
    Z = reducer.fit_transform(X_scaled)
    _plot_embedding(
        Z, metadata, method="UMAP", prefix="umap",
        subtitle=f"n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}",
    )
    _print_silhouette(Z, metadata, "UMAP")
    return Z


# ---------------------------------------------------------------------------
# Shared plotting
# ---------------------------------------------------------------------------

def _plot_embedding(
    Z: np.ndarray,
    metadata: pd.DataFrame,
    method: str,
    prefix: str,
    subtitle: str = "",
) -> None:
    label_specs: list[tuple[str, str, dict | None]] = []

    if "phase_label" in metadata.columns:
        label_specs.append(("phase_label", "Phase", PHASE_PALETTE))
    elif PHASE_COL in metadata.columns:
        label_specs.append((PHASE_COL, "Phase", None))

    col = "cohort_group" if "cohort_group" in metadata.columns else (
        COHORT_COL if COHORT_COL in metadata.columns else None
    )
    if col:
        label_specs.append((col, "Cohort", None))

    if PUZZLER_COL in metadata.columns:
        label_specs.append((PUZZLER_COL, "Role", None))

    if SUBJECT_COL in metadata.columns:
        label_specs.append((SUBJECT_COL, "Subject", None))

    if not label_specs:
        return

    n = len(label_specs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    df_plot = metadata.copy()
    df_plot["dim1"] = Z[:, 0]
    df_plot["dim2"] = Z[:, 1]

    for ax, (hue_col, label, palette) in zip(axes, label_specs):
        kw: dict = dict(data=df_plot, x="dim1", y="dim2", hue=hue_col, ax=ax,
                        alpha=0.75, s=50, linewidth=0)
        if palette:
            existing = set(df_plot[hue_col].dropna().unique())
            pal = {k: v for k, v in palette.items() if k in existing}
            if pal:
                kw["palette"] = pal
        sns.scatterplot(**kw)
        title = f"{method} — coloured by {label}"
        if subtitle:
            title += f"\n({subtitle})"
        ax.set_title(title)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        legend = ax.get_legend()
        if legend:
            legend.set_title(label)
            if hue_col == SUBJECT_COL:
                legend.remove()   # 26 subjects makes an unreadable legend

    plt.tight_layout()
    path = FIGURES_DIR / f"{prefix}_projections.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Silhouette score on phase labels
# ---------------------------------------------------------------------------

def _print_silhouette(Z: np.ndarray, metadata: pd.DataFrame, method: str) -> None:
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if label_col not in metadata.columns:
        return
    labels = metadata[label_col].values
    if len(np.unique(labels)) < 2:
        return
    score = silhouette_score(Z, labels)
    print(f"  Silhouette score (phase, {method}): {score:.3f}")
