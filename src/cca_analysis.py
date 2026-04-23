import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA

from src.config import (
    FIGURES_DIR, CCA_N_COMPONENTS,
    HR_PREFIXES, EDA_PREFIXES, TEMP_PREFIXES,
    PHASE_COL, PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)


def run_cca(
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
) -> None:
    """
    CCA between HR+TEMP features and EDA features.
    These represent two physiologically distinct sensing modalities:
    cardiovascular/thermoregulatory vs electrodermal activity.
    """
    hr_temp_idx = [
        i for i, f in enumerate(feature_names)
        if any(f.startswith(p) for p in HR_PREFIXES + TEMP_PREFIXES)
    ]
    eda_idx = [
        i for i, f in enumerate(feature_names)
        if any(f.startswith(p) for p in EDA_PREFIXES)
    ]

    if not hr_temp_idx or not eda_idx:
        print(
            "  [!] Skipping CCA — could not split features into HR/TEMP vs EDA groups.\n"
            f"  HR+TEMP prefixes searched: {HR_PREFIXES + TEMP_PREFIXES}\n"
            f"  EDA prefixes searched: {EDA_PREFIXES}\n"
            f"  First 15 feature names: {feature_names[:15]}\n"
            "  Adjust HR_PREFIXES / EDA_PREFIXES in src/config.py to match your data."
        )
        return

    X1 = X_scaled[:, hr_temp_idx]   # HR + TEMP
    X2 = X_scaled[:, eda_idx]       # EDA

    n_components = min(CCA_N_COMPONENTS, X1.shape[1], X2.shape[1])
    cca = CCA(n_components=n_components, max_iter=1000)
    Z1, Z2 = cca.fit_transform(X1, X2)

    correlations = [float(np.corrcoef(Z1[:, i], Z2[:, i])[0, 1]) for i in range(n_components)]
    print(f"  Canonical correlations: {[f'{c:.3f}' for c in correlations]}")

    hr_temp_names = [feature_names[i] for i in hr_temp_idx]
    eda_names = [feature_names[i] for i in eda_idx]

    _plot_canonical_scatter(Z1, Z2, metadata, correlations)
    _plot_loadings(cca, hr_temp_names, eda_names, n_components)


# ---------------------------------------------------------------------------
# Scatter of canonical variates
# ---------------------------------------------------------------------------

def _plot_canonical_scatter(
    Z1: np.ndarray,
    Z2: np.ndarray,
    metadata: pd.DataFrame,
    correlations: list[float],
) -> None:
    n = len(correlations)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    hue_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL

    for i, ax in enumerate(axes):
        df_plot = metadata.copy()
        df_plot["CV_HR"] = Z1[:, i]
        df_plot["CV_EDA"] = Z2[:, i]

        kw: dict = dict(data=df_plot, x="CV_HR", y="CV_EDA", hue=hue_col, ax=ax, alpha=0.75, s=55)
        if hue_col == "phase_label":
            existing = set(df_plot[hue_col].dropna().unique())
            pal = {k: v for k, v in PHASE_PALETTE.items() if k in existing}
            if pal:
                kw["palette"] = pal
        sns.scatterplot(**kw)

        r = correlations[i]
        ax.set_title(f"CCA Component {i+1}  (r = {r:.3f})")
        ax.set_xlabel(f"HR+TEMP Canonical Variate {i+1}")
        ax.set_ylabel(f"EDA Canonical Variate {i+1}")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.axvline(0, color="grey", lw=0.5, ls="--")
        ax.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    _save(fig, "cca_scatter.png")


# ---------------------------------------------------------------------------
# Feature loading bar charts
# ---------------------------------------------------------------------------

def _plot_loadings(
    cca: CCA,
    hr_temp_names: list[str],
    eda_names: list[str],
    n_components: int,
) -> None:
    cv_labels = [f"CV{i+1}" for i in range(n_components)]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, max(len(hr_temp_names), len(eda_names)) * 0.35 + 2)))

    for ax, names, weights, title in [
        (axes[0], hr_temp_names, cca.x_weights_, "HR + TEMP Feature Loadings"),
        (axes[1], eda_names, cca.y_weights_, "EDA Feature Loadings"),
    ]:
        df = pd.DataFrame(
            weights[:, :n_components],
            index=names,
            columns=cv_labels,
        )
        colors = ["#4C72B0", "#C44E52"][:n_components]
        df.plot.barh(ax=ax, color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Loading Weight")
        ax.legend(title="Component")

    plt.tight_layout()
    _save(fig, "cca_loadings.png")


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
