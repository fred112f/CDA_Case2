import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.config import (
    FIGURES_DIR, PHASE_COL, PUZZLER_COL,
    PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)


def run_role_analysis(
    Z_pca: np.ndarray,
    metadata: pd.DataFrame,
    n_pcs: int = 4,
) -> None:
    if PUZZLER_COL not in metadata.columns:
        print(f"  Column '{PUZZLER_COL}' not found — skipping role analysis.")
        return

    phase_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    n_pcs = min(n_pcs, Z_pca.shape[1])

    df = metadata[[PUZZLER_COL] + ([phase_col] if phase_col in metadata.columns else [])].copy()
    for i in range(n_pcs):
        df[f"PC{i+1}"] = Z_pca[:, i]

    _plot_scatter(Z_pca, df, phase_col)
    _plot_boxplots(df, phase_col, n_pcs)
    _print_mannwhitney(df, n_pcs)


def _plot_scatter(Z_pca: np.ndarray, df: pd.DataFrame, phase_col: str) -> None:
    """PC1 vs PC2 with role as marker shape and phase as colour."""
    roles = sorted(df[PUZZLER_COL].dropna().unique())
    markers = ["o", "^", "s", "D"]
    role_marker = dict(zip(roles, markers))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each role group separately so markers differ
    for role in roles:
        sub = df[df[PUZZLER_COL] == role].copy()
        hue_col = phase_col if phase_col in sub.columns else None
        existing = set(sub[hue_col].dropna().unique()) if hue_col else set()
        pal = {k: v for k, v in PHASE_PALETTE.items() if k in existing} if hue_col else None

        sns.scatterplot(
            data=sub, x="PC1", y="PC2",
            hue=hue_col, palette=pal or None,
            marker=role_marker[role], ax=ax,
            alpha=0.72, s=65, linewidth=0,
            legend=(role == roles[0]),
        )

    # Role legend (marker shapes)
    role_handles = [
        mlines.Line2D([], [], marker=role_marker[r], color="grey",
                      linestyle="None", ms=9, label=str(r))
        for r in roles
    ]
    role_legend = ax.legend(handles=role_handles, title="Role", loc="lower right", fontsize=8)
    ax.add_artist(role_legend)
    ax.legend(title="Phase", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax.set_title("PC Space by Role (marker shape) and Phase (colour)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    plt.tight_layout()
    _save(fig, "role_pca_scatter.png")


def _plot_boxplots(df: pd.DataFrame, phase_col: str, n_pcs: int) -> None:
    """Boxplot of PC scores split by role, with phase on x-axis."""
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]

    has_phase = phase_col in df.columns
    fig, axes = plt.subplots(1, n_pcs, figsize=(4.5 * n_pcs, 5))
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, pc_cols):
        if has_phase:
            sns.boxplot(
                data=df, x=phase_col, y=pc, hue=PUZZLER_COL,
                ax=ax, palette="Set2", width=0.6,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
            ax.get_legend().remove() if ax.get_legend() else None
        else:
            sns.boxplot(data=df, x=PUZZLER_COL, y=pc, ax=ax, palette="Set2", width=0.5)
            ax.set_xlabel("Role")

        ax.set_title(pc)
        ax.set_ylabel("PC Score")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels() if not has_phase else (
        plt.Line2D([], []),  # fallback
        [],
    )
    if has_phase:
        roles = df[PUZZLER_COL].dropna().unique()
        import matplotlib.patches as mpatches
        import matplotlib.cm as cm
        colors = sns.color_palette("Set2", len(roles))
        handles = [mpatches.Patch(color=c, label=r) for c, r in zip(colors, roles)]
        fig.legend(handles=handles, title="Role", loc="lower center",
                   ncol=len(roles), bbox_to_anchor=(0.5, -0.05), fontsize=8)

    fig.suptitle("PC Scores by Role and Phase", fontsize=12)
    plt.tight_layout()
    _save(fig, "role_boxplots.png")


def _print_mannwhitney(df: pd.DataFrame, n_pcs: int) -> None:
    roles = df[PUZZLER_COL].dropna().unique()
    if len(roles) != 2:
        return
    r1, r2 = roles
    print(f"\n  Mann-Whitney U test ({r1} vs {r2}):")
    for i in range(n_pcs):
        col = f"PC{i+1}"
        g1 = df.loc[df[PUZZLER_COL] == r1, col].dropna()
        g2 = df.loc[df[PUZZLER_COL] == r2, col].dropna()
        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        sig = "  *" if p < 0.05 else ""
        print(f"    {col}: U = {stat:.0f},  p = {p:.4f}{sig}")


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
