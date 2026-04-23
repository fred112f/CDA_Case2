import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA

from src.config import (
    FIGURES_DIR, PHASE_COL, SUBJECT_COL, ROUND_COL,
    PUZZLER_COL, FIGURE_DPI, FIGURE_FORMAT,
)

_PHASE_ORDER: dict[str, int] = {
    "Pre-puzzle (Rest)": 0,
    "Puzzle (Stress)": 1,
    "Post-puzzle (Recovery)": 2,
}

_DELTA_PALETTE: dict[str, str] = {
    "Stress response": "#C44E52",
    "Recovery response": "#55A868",
}

_TOP_N = 15  # features shown in the mean-delta bar chart


def run_delta_analysis(
    X_scaled: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
) -> None:
    """
    Compute within-subject phase deltas per (subject, round):
        stress_delta    = phase2 − phase1
        recovery_delta  = phase3 − phase1

    These differences cancel individual baselines entirely and directly
    capture the physiological stress and recovery transitions.
    """
    label_col = "phase_label" if "phase_label" in metadata.columns else PHASE_COL
    required = [SUBJECT_COL, ROUND_COL, label_col]
    missing = [c for c in required if c not in metadata.columns]
    if missing:
        print(f"  Missing columns {missing} — skipping delta analysis.")
        return

    keep = [SUBJECT_COL, ROUND_COL, label_col]
    for col in (PUZZLER_COL, "cohort_group"):
        if col in metadata.columns:
            keep.append(col)

    df = metadata[keep].copy()
    df["_phase_ord"] = df[label_col].map(_PHASE_ORDER)

    feat_arr = feature_names  # shorter alias
    for i, name in enumerate(feat_arr):
        df[name] = X_scaled[:, i]

    df = df.dropna(subset=["_phase_ord"])

    stress_rows: list[np.ndarray] = []
    recovery_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for (subj, rnd), grp in df.groupby([SUBJECT_COL, ROUND_COL]):
        phases_present = set(grp["_phase_ord"].astype(int).tolist())
        if phases_present != {0, 1, 2}:
            continue

        grp_s = grp.sort_values("_phase_ord")
        p1 = grp_s[grp_s["_phase_ord"] == 0][feat_arr].values[0]
        p2 = grp_s[grp_s["_phase_ord"] == 1][feat_arr].values[0]
        p3 = grp_s[grp_s["_phase_ord"] == 2][feat_arr].values[0]

        stress_rows.append(p2 - p1)
        recovery_rows.append(p3 - p1)

        row: dict = {SUBJECT_COL: subj, ROUND_COL: rnd}
        for col in (PUZZLER_COL, "cohort_group"):
            if col in grp.columns:
                row[col] = grp[col].iloc[0]
        meta_rows.append(row)

    if not stress_rows:
        print("  No complete phase triplets found — skipping delta analysis.")
        return

    X_stress = np.array(stress_rows)
    X_recovery = np.array(recovery_rows)
    meta_df = pd.DataFrame(meta_rows)

    print(f"  Delta analysis: {len(stress_rows)} complete subject×round triplets")

    _plot_mean_deltas(X_stress, X_recovery, feature_names)
    _run_delta_pca(X_stress, X_recovery, feature_names, meta_df)
    _print_significant_deltas(X_stress, X_recovery, feature_names)


def _plot_mean_deltas(
    X_stress: np.ndarray,
    X_recovery: np.ndarray,
    feature_names: list[str],
    top_n: int = _TOP_N,
) -> None:
    mean_s = X_stress.mean(axis=0)
    sem_s = X_stress.std(axis=0) / np.sqrt(X_stress.shape[0])
    mean_r = X_recovery.mean(axis=0)
    sem_r = X_recovery.std(axis=0) / np.sqrt(X_recovery.shape[0])

    rank = np.argsort(np.abs(mean_s))[::-1][:top_n]
    # bottom-to-top display
    rank_rev = rank[::-1]
    names = [feature_names[i] for i in rank_rev]
    m_s, e_s = mean_s[rank_rev], sem_s[rank_rev]
    m_r, e_r = mean_r[rank_rev], sem_r[rank_rev]

    y = np.arange(len(names))
    h = 0.35
    ekw = {"elinewidth": 0.8, "capsize": 2}

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    ax.barh(y + h / 2, m_s, h, xerr=e_s,
            color=_DELTA_PALETTE["Stress response"], alpha=0.85,
            label="Stress response", error_kw=ekw)
    ax.barh(y - h / 2, m_r, h, xerr=e_r,
            color=_DELTA_PALETTE["Recovery response"], alpha=0.85,
            label="Recovery response", error_kw=ekw)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean delta  (within-subject z-score units) ± SEM")
    ax.set_title(
        f"Phase Delta Profile — Top {top_n} features by |stress response|\n"
        "Positive = higher during that phase than at rest (phase 1 baseline)"
    )
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "delta_mean_profile.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")


def _run_delta_pca(
    X_stress: np.ndarray,
    X_recovery: np.ndarray,
    feature_names: list[str],
    meta_df: pd.DataFrame,
) -> None:
    X_all = np.vstack([X_stress, X_recovery])
    delta_labels = (["Stress response"] * len(X_stress) +
                    ["Recovery response"] * len(X_recovery))

    n_comp = min(2, X_all.shape[1], X_all.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    Z = pca.fit_transform(X_all)
    evr = pca.explained_variance_ratio_

    df_plot = pd.concat([meta_df, meta_df], ignore_index=True).copy()
    df_plot["Delta type"] = delta_labels
    df_plot["PC1"] = Z[:, 0]
    df_plot["PC2"] = Z[:, 1] if Z.shape[1] > 1 else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2", hue="Delta type",
        palette=_DELTA_PALETTE, ax=ax, alpha=0.75, s=60, linewidth=0,
    )
    ax.set_title("PCA on Phase-Change Vectors\n"
                 "(stress = phase2−phase1,  recovery = phase3−phase1)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f} %)" if Z.shape[1] > 1 else "PC2")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.legend(title="Delta type", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "delta_pca.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")

    if PUZZLER_COL in meta_df.columns and n_comp >= 2:
        _plot_delta_pca_role(Z[:len(meta_df)], meta_df, evr)


def _plot_delta_pca_role(
    Z_stress: np.ndarray,
    meta_df: pd.DataFrame,
    evr: np.ndarray,
) -> None:
    df = meta_df.copy()
    df["PC1"] = Z_stress[:, 0]
    df["PC2"] = Z_stress[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df, x="PC1", y="PC2", hue=PUZZLER_COL,
        palette="Set2", ax=ax, alpha=0.75, s=60, linewidth=0,
    )
    ax.set_title("Stress Response PCA — coloured by Role\n(phase2 − phase1)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f} %)")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
    ax.legend(title="Role", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = FIGURES_DIR / "delta_pca_role.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", format=FIGURE_FORMAT)
    plt.close(fig)
    print(f"  Saved: {path}")


def _print_significant_deltas(
    X_stress: np.ndarray,
    X_recovery: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.05,
) -> None:
    n = len(feature_names)
    alpha_corr = alpha / (2 * n)  # Bonferroni for 2 × n tests

    sig_s: list[tuple[str, float, float]] = []
    sig_r: list[tuple[str, float, float]] = []

    for i, name in enumerate(feature_names):
        _, p_s = ttest_1samp(X_stress[:, i], popmean=0)
        _, p_r = ttest_1samp(X_recovery[:, i], popmean=0)
        if p_s < alpha_corr:
            sig_s.append((name, float(X_stress[:, i].mean()), p_s))
        if p_r < alpha_corr:
            sig_r.append((name, float(X_recovery[:, i].mean()), p_r))

    print(f"\n  Significant stress-response features (Bonferroni α = {alpha_corr:.4f}):")
    if sig_s:
        for name, mean, p in sorted(sig_s, key=lambda x: abs(x[1]), reverse=True):
            print(f"    {name}: mean_delta = {mean:+.3f},  p = {p:.4f}")
    else:
        print("    None (weak or variable stress responses across subjects).")

    print(f"\n  Significant recovery features (Bonferroni α = {alpha_corr:.4f}):")
    if sig_r:
        for name, mean, p in sorted(sig_r, key=lambda x: abs(x[1]), reverse=True):
            print(f"    {name}: mean_delta = {mean:+.3f},  p = {p:.4f}")
    else:
        print("    None.")
