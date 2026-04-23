import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import (
    FIGURES_DIR, PHASE_COL,
    PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)


def run_lda_comparison(
    X_scaled: np.ndarray,
    metadata: pd.DataFrame,
    pca_silhouette_2d: float | None = None,
) -> None:
    """
    Fit LDA with phase labels (supervised) and compare its separation
    to the unsupervised PCA result.  Cross-validated accuracy gives the
    best-case predictability of phase from biosignals.
    """
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if label_col not in metadata.columns:
        print("  No phase labels — skipping LDA.")
        return

    labels = metadata[label_col].values
    if len(np.unique(labels)) < 2:
        return

    lda = LinearDiscriminantAnalysis()
    Z_lda = lda.fit_transform(X_scaled, labels)

    sil_lda = silhouette_score(Z_lda, labels)
    print(f"  LDA  silhouette (phase): {sil_lda:.3f}")
    if pca_silhouette_2d is not None:
        print(f"  PCA  silhouette (phase): {pca_silhouette_2d:.3f}  "
              f"→ supervised gain: {sil_lda - pca_silhouette_2d:+.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(lda, X_scaled, labels, cv=cv, scoring="accuracy")
    chance = 1.0 / len(np.unique(labels))
    print(f"  LDA 5-fold CV accuracy : {acc.mean():.3f} ± {acc.std():.3f}  "
          f"(chance = {chance:.3f})")

    _plot_projection(Z_lda, metadata, label_col, sil_lda, acc.mean(), chance)


def _plot_projection(
    Z_lda: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str,
    sil: float,
    acc: float,
    chance: float,
) -> None:
    df = metadata.copy()
    df["LD1"] = Z_lda[:, 0]
    df["LD2"] = Z_lda[:, 1] if Z_lda.shape[1] > 1 else 0.0

    existing = set(df[label_col].dropna().unique())
    pal = {k: v for k, v in PHASE_PALETTE.items() if k in existing}

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df, x="LD1", y="LD2", hue=label_col,
        palette=pal or None, ax=ax, alpha=0.75, s=55, linewidth=0,
    )
    ax.set_title(
        f"LDA Projection  (supervised, phase labels)\n"
        f"Silhouette = {sil:.3f}  |  5-fold CV acc = {acc:.3f}  (chance = {chance:.3f})"
    )
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "lda_projection.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
