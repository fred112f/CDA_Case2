import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    FIGURES_DIR, PHASE_COL, PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)

_SIGNAL_GROUPS: list[tuple[str, str]] = [
    ("HR", "#4C72B0"),
    ("TEMP", "#DD8452"),
    ("EDA_TD_P", "#55A868"),
    ("EDA_TD_T", "#C44E52"),
]


def run_feature_importance(
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
    top_n: int = 20,
) -> np.ndarray:
    """
    One-way ANOVA F-test for each feature across phase labels.
    Saves a ranked importance chart and a PCA plot using only the top features.
    Returns a boolean mask over feature_names selecting the top_n.
    """
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if label_col not in metadata.columns:
        print("  No phase labels — skipping feature importance.")
        return np.ones(len(feature_names), dtype=bool)

    labels = metadata[label_col].values
    if len(np.unique(labels)) < 2:
        return np.ones(len(feature_names), dtype=bool)

    F, pvals = f_classif(X_scaled, labels)

    order = np.argsort(F)[::-1]
    top_n = min(top_n, len(feature_names))
    top_mask = np.zeros(len(feature_names), dtype=bool)
    top_mask[order[:top_n]] = True

    _plot_importance(F, pvals, feature_names, order, top_n)
    _pca_selected(
        X_scaled[:, top_mask],
        [feature_names[i] for i in order[:top_n]],
        metadata,
        label_col,
    )

    print(f"\n  Top {min(10, top_n)} phase-discriminating features (ANOVA F-test):")
    for i in order[:10]:
        sig = "*" if pvals[i] < 0.05 else ""
        print(f"    {feature_names[i]}: F = {F[i]:.2f},  p = {pvals[i]:.4f}{sig}")

    return top_mask


def _group_color(name: str) -> str:
    for prefix, color in _SIGNAL_GROUPS:
        if name.startswith(prefix):
            return color
    return "#7f7f7f"


def _plot_importance(
    F: np.ndarray,
    pvals: np.ndarray,
    feature_names: list[str],
    order: np.ndarray,
    top_n: int,
) -> None:
    show = order[:top_n]
    names_rev = [feature_names[i] for i in show[::-1]]  # bottom → top
    values_rev = F[show[::-1]]
    colors_rev = [_group_color(n) for n in names_rev]
    sig_rev = pvals[show[::-1]] < 0.05

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.42)))
    ax.barh(range(len(names_rev)), values_rev, color=colors_rev,
            alpha=0.85, edgecolor="white")

    x_max = float(values_rev.max()) if len(values_rev) else 1.0
    for idx, (val, is_sig) in enumerate(zip(values_rev, sig_rev)):
        if is_sig:
            ax.text(val + x_max * 0.01, idx, "*", va="center", fontsize=9)

    ax.set_yticks(range(len(names_rev)))
    ax.set_yticklabels(names_rev, fontsize=8)
    ax.set_xlabel("ANOVA F-statistic  (phase discrimination)")
    ax.set_title(f"Top {top_n} Phase-Discriminating Features\n(* = p < 0.05 uncorrected)")

    handles = [mpatches.Patch(color=c, label=g) for g, c in _SIGNAL_GROUPS]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")


def _pca_selected(
    X_sel: np.ndarray,
    selected_names: list[str],
    metadata: pd.DataFrame,
    label_col: str,
) -> None:
    n_comp = min(2, X_sel.shape[1], X_sel.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    Z = pca.fit_transform(X_sel)
    evr = pca.explained_variance_ratio_

    labels = metadata[label_col].values
    sil = None
    if n_comp >= 2 and len(np.unique(labels)) >= 2:
        sil = silhouette_score(Z[:, :2], labels)
        print(f"  PCA (top {len(selected_names)} features) silhouette: {sil:.3f}")

    df_plot = metadata[[label_col]].copy()
    df_plot["PC1"] = Z[:, 0]
    df_plot["PC2"] = Z[:, 1] if Z.shape[1] > 1 else 0.0
    existing = set(df_plot[label_col].dropna().unique())
    pal = {k: v for k, v in PHASE_PALETTE.items() if k in existing}

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2", hue=label_col,
        palette=pal or None, ax=ax, alpha=0.75, s=55, linewidth=0,
    )
    title = f"PCA on top {len(selected_names)} phase-discriminating features"
    subtitle_parts = []
    if Z.shape[1] >= 2:
        subtitle_parts.append(f"PC1+PC2 = {(evr[0]+evr[1])*100:.1f}%")
    if sil is not None:
        subtitle_parts.append(f"Silhouette = {sil:.3f}")
    if subtitle_parts:
        title += "\n" + "  |  ".join(subtitle_parts)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f} %)" if Z.shape[1] > 1 else "PC2 (n/a)")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "pca_selected_features.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
