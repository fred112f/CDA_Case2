import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    FIGURES_DIR, PCA_N_COMPONENTS, PCA_BIPLOT_N_FEATURES,
    PHASE_COL, COHORT_COL, SUBJECT_COL, PUZZLER_COL,
    PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)


def run_pca_analysis(
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
) -> tuple[PCA, np.ndarray, float | None]:
    """Returns (pca, Z, silhouette_2d) — silhouette_2d is None if no phase labels."""
    n_components = min(PCA_N_COMPONENTS, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)

    plot_feature_correlations(X_scaled, feature_names)
    _plot_scree(pca)
    _plot_projections(Z, metadata, pca)
    _plot_biplot(Z, pca, feature_names)
    _print_loadings(pca, feature_names)
    sil_2d = _print_silhouette(Z, metadata)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n80 = int(np.searchsorted(cumvar, 0.80)) + 1
    print(
        f"  PC1+PC2 explain {cumvar[1]*100:.1f}% of variance  |  "
        f"{n80} components needed for 80%"
    )
    return pca, Z, sil_2d


# ---------------------------------------------------------------------------
# Feature correlation heatmap  (motivates PCA)
# ---------------------------------------------------------------------------

def plot_feature_correlations(X_scaled: np.ndarray, feature_names: list[str]) -> None:
    # Sort features by signal group so the block structure is visible
    groups = ["HR", "TEMP", "EDA_TD_P", "EDA_TD_T"]
    ordered: list[str] = []
    for g in groups:
        ordered.extend(f for f in feature_names if f.startswith(g) and f not in ordered)
    ordered.extend(f for f in feature_names if f not in ordered)  # any remainder

    idx = [feature_names.index(f) for f in ordered]
    corr = np.corrcoef(X_scaled[:, idx].T)

    # Build group boundary lines
    boundaries: list[int] = []
    prev = None
    for f in ordered:
        grp = next((g for g in groups if f.startswith(g)), "other")
        if grp != prev:
            boundaries.append(ordered.index(f))
            prev = grp

    n = len(ordered)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.28), max(12, n * 0.26)))
    sns.heatmap(
        corr,
        xticklabels=ordered, yticklabels=ordered,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.0, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    for b in boundaries[1:]:
        ax.axhline(b, color="black", lw=1.2)
        ax.axvline(b, color="black", lw=1.2)

    ax.set_title("Feature Correlation Matrix (grouped by signal type)")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)
    plt.tight_layout()
    _save(fig, "feature_correlations.png")


# ---------------------------------------------------------------------------
# Scree plot
# ---------------------------------------------------------------------------

def _plot_scree(pca: PCA) -> None:
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(x, evr * 100, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Scree Plot")
    ax1.set_xticks(x)

    ax2.plot(x, cumvar * 100, "o-", color="#C44E52", lw=2, ms=6)
    for threshold, ls, label in [(80, "--", "80 %"), (90, ":", "90 %")]:
        ax2.axhline(threshold, color="grey", linestyle=ls, lw=1, label=label)
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(x)
    ax2.set_ylim(0, 105)
    ax2.legend()

    plt.tight_layout()
    _save(fig, "pca_scree.png")


# ---------------------------------------------------------------------------
# 2-D scatter projections — 2 rows (PC1 vs PC2, PC1 vs PC3)
# ---------------------------------------------------------------------------

def _plot_projections(Z: np.ndarray, metadata: pd.DataFrame, pca: PCA) -> None:
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

    evr = pca.explained_variance_ratio_
    pc_pairs = [
        (0, 1, f"PC1 ({evr[0]*100:.1f} %)", f"PC2 ({evr[1]*100:.1f} %)"),
        (0, 2, f"PC1 ({evr[0]*100:.1f} %)", f"PC3 ({evr[2]*100:.1f} %)"),
    ]

    n_cols = len(label_specs)
    n_rows = len(pc_pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    # Ensure 2-D axes array
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    df_plot = metadata.copy()
    for pci, pcj, xlbl, ylbl in pc_pairs:
        df_plot[f"PC{pci+1}"] = Z[:, pci]
        df_plot[f"PC{pcj+1}"] = Z[:, pcj]

    for row_i, (pci, pcj, xlbl, ylbl) in enumerate(pc_pairs):
        xk, yk = f"PC{pci+1}", f"PC{pcj+1}"
        for col_j, (hue_col, label, palette) in enumerate(label_specs):
            ax = axes[row_i, col_j]
            kw: dict = dict(data=df_plot, x=xk, y=yk, hue=hue_col, ax=ax,
                            alpha=0.75, s=50, linewidth=0)
            if palette:
                existing = set(df_plot[hue_col].dropna().unique())
                pal = {k: v for k, v in palette.items() if k in existing}
                if pal:
                    kw["palette"] = pal
            sns.scatterplot(**kw)
            ax.set_title(f"PCA — coloured by {label}")
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            ax.axhline(0, color="grey", lw=0.4, ls="--")
            ax.axvline(0, color="grey", lw=0.4, ls="--")
            legend = ax.get_legend()
            if legend:
                legend.set_title(label)
                # Suppress legend for Subject (too many entries)
                if hue_col == SUBJECT_COL:
                    legend.remove()

    plt.tight_layout()
    _save(fig, "pca_projections.png")


# ---------------------------------------------------------------------------
# Biplot
# ---------------------------------------------------------------------------

def _plot_biplot(Z: np.ndarray, pca: PCA, feature_names: list[str]) -> None:
    n_show = min(PCA_BIPLOT_N_FEATURES, len(feature_names))
    loadings = pca.components_.T          # (n_features, n_components)

    magnitudes = np.hypot(loadings[:, 0], loadings[:, 1])
    top_idx = np.argsort(magnitudes)[-n_show:]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Z[:, 0], Z[:, 1], alpha=0.25, s=25, color="steelblue")

    scale = 0.4 * max(
        float(Z[:, 0].max() - Z[:, 0].min()),
        float(Z[:, 1].max() - Z[:, 1].min()),
    ) / magnitudes[top_idx].max()

    for i in top_idx:
        dx, dy = loadings[i, 0] * scale, loadings[i, 1] * scale
        ax.annotate(
            "",
            xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
        )
        ax.text(dx * 1.12, dy * 1.12, feature_names[i],
                fontsize=8, color="crimson", ha="center", va="center")

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f} %)")
    ax.set_title(f"PCA Biplot — top {n_show} features by loading magnitude")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")

    plt.tight_layout()
    _save(fig, "pca_biplot.png")


# ---------------------------------------------------------------------------
# Silhouette score on phase labels
# ---------------------------------------------------------------------------

def _print_silhouette(Z: np.ndarray, metadata: pd.DataFrame) -> float | None:
    """Prints and returns the 2-D silhouette score, or None if unavailable."""
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if label_col not in metadata.columns:
        return None
    labels = metadata[label_col].values
    if len(np.unique(labels)) < 2:
        return None
    score_2d = silhouette_score(Z[:, :2], labels)
    score_full = silhouette_score(Z, labels)
    print(f"  Silhouette score (phase) — PC1-2: {score_2d:.3f}  |  all PCs: {score_full:.3f}")
    return score_2d


# ---------------------------------------------------------------------------
# Print top loadings table
# ---------------------------------------------------------------------------

def _print_loadings(pca: PCA, feature_names: list[str], top_n: int = 5) -> None:
    print("\n  Top feature loadings per component:")
    for i, comp in enumerate(pca.components_[:3]):
        top = np.argsort(np.abs(comp))[-top_n:][::-1]
        row = ", ".join(f"{feature_names[j]} ({comp[j]:+.3f})" for j in top)
        print(f"    PC{i+1}: {row}")


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
