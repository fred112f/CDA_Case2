import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap

from src.config import (
    FIGURES_DIR, PHASE_COL, SUBJECT_COL,
    PHASE_PALETTE, FIGURE_DPI, FIGURE_FORMAT,
)

_PHASE_ORDER = ["Pre-puzzle (Rest)", "Puzzle (Stress)", "Post-puzzle (Recovery)"]


def run_subject_trajectories(
    Z_pca: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    phase_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    if phase_col not in metadata.columns or SUBJECT_COL not in metadata.columns:
        print("  Missing phase or subject column — skipping trajectories.")
        return

    df = metadata[[SUBJECT_COL, phase_col]].copy()
    df["PC1"] = Z_pca[:, 0]
    df["PC2"] = Z_pca[:, 1]

    traj = (
        df.groupby([SUBJECT_COL, phase_col])[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )

    _plot_individual(traj, phase_col)
    _plot_mean(traj, phase_col)


def _plot_individual(traj: pd.DataFrame, phase_col: str) -> None:
    subjects = sorted(traj[SUBJECT_COL].unique())
    cmap = get_cmap("tab20", len(subjects))

    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, subj in enumerate(subjects):
        grp = (
            traj[traj[SUBJECT_COL] == subj]
            .set_index(phase_col)
            .reindex(_PHASE_ORDER)
            .dropna()
        )
        if len(grp) < 2:
            continue

        color = cmap(idx)
        ax.plot(grp["PC1"], grp["PC2"], "o-", color=color, alpha=0.45, lw=1.2, ms=5)

        # Arrow on the stress → recovery segment
        if len(grp) == 3:
            ax.annotate(
                "",
                xy=(grp["PC1"].iloc[2], grp["PC2"].iloc[2]),
                xytext=(grp["PC1"].iloc[1], grp["PC2"].iloc[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    # Phase legend via dummy scatter
    for phase, color in PHASE_PALETTE.items():
        ax.scatter([], [], color=color, s=50, label=phase)
    ax.legend(title="Phase", fontsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Per-Subject Trajectories Through PC Space\n(◦ = Rest → Stress → Recovery)")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    plt.tight_layout()
    _save(fig, "trajectories_individual.png")


def _plot_mean(traj: pd.DataFrame, phase_col: str) -> None:
    agg = (
        traj.groupby(phase_col)[["PC1", "PC2"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    agg.columns = [phase_col, "PC1_mean", "PC1_sem", "PC2_mean", "PC2_sem"]
    agg = agg.set_index(phase_col).reindex(_PHASE_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = [PHASE_PALETTE.get(p, "grey") for p in agg.index]

    for i, (phase, row) in enumerate(agg.iterrows()):
        ax.errorbar(
            row["PC1_mean"], row["PC2_mean"],
            xerr=row["PC1_sem"], yerr=row["PC2_sem"],
            fmt="o", color=colors[i], ms=14, capsize=5, lw=2, label=phase,
            zorder=3,
        )

    # Arrows between consecutive mean positions
    for i in range(len(agg) - 1):
        ax.annotate(
            "",
            xy=(agg["PC1_mean"].iloc[i + 1], agg["PC2_mean"].iloc[i + 1]),
            xytext=(agg["PC1_mean"].iloc[i], agg["PC2_mean"].iloc[i]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.8),
        )

    ax.set_xlabel("PC1 (mean ± SEM across subjects)")
    ax.set_ylabel("PC2 (mean ± SEM across subjects)")
    ax.set_title("Group-Mean Trajectory Through PC Space")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.legend()
    plt.tight_layout()
    _save(fig, "trajectories_mean.png")


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")
