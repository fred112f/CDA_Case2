import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.config import FIGURES_DIR, PHASE_COL, FIGURE_DPI, FIGURE_FORMAT


def run_permutation_test(
    Z_pca: np.ndarray,
    metadata: pd.DataFrame,
    n_permutations: int = 1000,
) -> None:
    """
    Test whether the observed silhouette score (PC1-2, phase labels) is
    significantly above chance by comparing to a null distribution built
    from randomly shuffled labels.
    """
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if label_col not in metadata.columns:
        print("  No phase labels — skipping permutation test.")
        return

    labels = metadata[label_col].values
    if len(np.unique(labels)) < 2:
        return

    Z2 = Z_pca[:, :2]
    observed = silhouette_score(Z2, labels)

    rng = np.random.default_rng(42)
    null = np.array([
        silhouette_score(Z2, rng.permutation(labels))
        for _ in range(n_permutations)
    ])

    p_value = float((null >= observed).mean())

    print(f"  Observed silhouette : {observed:.4f}")
    print(f"  Null mean ± std     : {null.mean():.4f} ± {null.std():.4f}")
    print(f"  p-value ({n_permutations} permutations): {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'not significant'} at α=0.05)")

    _plot_null_distribution(null, observed, p_value, n_permutations)


def _plot_null_distribution(
    null: np.ndarray,
    observed: float,
    p_value: float,
    n_permutations: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(null, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white",
            label=f"Null distribution\n(n={n_permutations} permutations)")
    ax.axvline(observed, color="#C44E52", lw=2.5,
               label=f"Observed = {observed:.4f}\np = {p_value:.4f}")

    # 95th percentile of null
    q95 = np.percentile(null, 95)
    ax.axvline(q95, color="grey", lw=1.5, ls="--", label=f"Null 95th pct = {q95:.4f}")

    ax.set_xlabel("Silhouette Score (phase labels, PC1–2)")
    ax.set_ylabel("Count")
    ax.set_title("Permutation Test — Phase Separation in PCA Space")
    ax.legend()
    plt.tight_layout()

    path = FIGURES_DIR / "permutation_test.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
