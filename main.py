"""
CDA Case II — Topic 2: The Representation Challenge
====================================================
Pipeline for dimensionality reduction and representation analysis
of the EmoPairCompete biosignal features.

Usage:
    uv run python main.py

Enable / disable individual steps in src/config.py under PIPELINE.
Outputs are saved to outputs/figures/.
"""

import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from src.config import FIGURES_DIR, HR_DATA_FILE, PIPELINE
from src.loader import load_hr_data, validate_columns
from src.preprocessing import prepare_features
from src.pca_analysis import run_pca_analysis
from src.manifold_analysis import run_tsne, run_umap
from src.cca_analysis import run_cca
from src.questionnaire_analysis import run_questionnaire_correlation
from src.trajectory_analysis import run_subject_trajectories
from src.lda_analysis import run_lda_comparison
from src.permutation_test import run_permutation_test
from src.role_analysis import run_role_analysis
from src.feature_importance import run_feature_importance
from src.delta_analysis import run_delta_analysis
from src.round_analysis import run_round_analysis


def _step(key: str, label: str) -> bool:
    enabled = PIPELINE.get(key, True)
    status = "ON " if enabled else "OFF"
    print(f"\n[{status}] {label}")
    return enabled


def main() -> None:
    print("=" * 60)
    print("CDA Case II — The Representation Challenge")
    print("=" * 60)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # ------------------------------------------------------------------ #
    # Load & preprocess  (always runs)
    # ------------------------------------------------------------------ #
    print("\n[LOAD] Loading data ...")
    df = load_hr_data(HR_DATA_FILE)
    validate_columns(df)

    print("\n[PREP] Preprocessing ...")
    X_scaled, feature_names, metadata = prepare_features(df)

    # ------------------------------------------------------------------ #
    # PCA
    # ------------------------------------------------------------------ #
    if _step("pca", "PCA  (scree · projections · biplot · feature correlations)"):
        pca, Z_pca, pca_sil_2d = run_pca_analysis(X_scaled, feature_names, metadata)
        results["pca"] = pca
        results["Z_pca"] = Z_pca
        results["pca_sil_2d"] = pca_sil_2d  # float | None, reused by LDA step

    # ------------------------------------------------------------------ #
    # Non-linear manifold learning
    # ------------------------------------------------------------------ #
    if _step("tsne", "t-SNE"):
        results["Z_tsne"] = run_tsne(X_scaled, metadata)

    if _step("umap", "UMAP"):
        results["Z_umap"] = run_umap(X_scaled, metadata)

    # ------------------------------------------------------------------ #
    # CCA
    # ------------------------------------------------------------------ #
    if _step("cca", "CCA  (HR+TEMP vs EDA)"):
        run_cca(X_scaled, feature_names, metadata)

    # ------------------------------------------------------------------ #
    # Questionnaire–PC correlation
    # ------------------------------------------------------------------ #
    if _step("questionnaire_correlation", "Questionnaire × PC correlation"):
        if "Z_pca" in results:
            run_questionnaire_correlation(results["Z_pca"], metadata)
        else:
            print("  [skip] requires PCA — set 'pca': True in PIPELINE")

    # ------------------------------------------------------------------ #
    # Per-subject trajectories through PC space
    # ------------------------------------------------------------------ #
    if _step("subject_trajectories", "Per-subject trajectories through PC space"):
        if "Z_pca" in results:
            run_subject_trajectories(results["Z_pca"], metadata)
        else:
            print("  [skip] requires PCA — set 'pca': True in PIPELINE")

    # ------------------------------------------------------------------ #
    # LDA  (supervised upper bound)
    # ------------------------------------------------------------------ #
    if _step("lda_comparison", "LDA comparison  (supervised upper bound)"):
        run_lda_comparison(
            X_scaled, metadata,
            pca_silhouette_2d=results.get("pca_sil_2d"),
        )

    # ------------------------------------------------------------------ #
    # Permutation test on silhouette
    # ------------------------------------------------------------------ #
    if _step("permutation_test", "Permutation test on silhouette score"):
        if "Z_pca" in results:
            run_permutation_test(results["Z_pca"], metadata)
        else:
            print("  [skip] requires PCA — set 'pca': True in PIPELINE")

    # ------------------------------------------------------------------ #
    # Role analysis  (Puzzler vs Instructor)
    # ------------------------------------------------------------------ #
    if _step("role_analysis", "Role analysis  (Puzzler vs Instructor)"):
        if "Z_pca" in results:
            run_role_analysis(results["Z_pca"], metadata)
        else:
            print("  [skip] requires PCA — set 'pca': True in PIPELINE")

    # ------------------------------------------------------------------ #
    # Feature importance  (ANOVA F-test + selected-feature PCA)
    # ------------------------------------------------------------------ #
    if _step("feature_importance", "Feature importance  (ANOVA F-test per feature)"):
        run_feature_importance(X_scaled, feature_names, metadata)

    # ------------------------------------------------------------------ #
    # Delta analysis  (within-subject phase change vectors)
    # ------------------------------------------------------------------ #
    if _step("delta_analysis", "Delta analysis  (phase2−phase1, phase3−phase1)"):
        run_delta_analysis(X_scaled, feature_names, metadata)

    # ------------------------------------------------------------------ #
    # Round analysis  (silhouette + PCA facets per round)
    # ------------------------------------------------------------------ #
    if _step("round_analysis", "Round analysis  (phase structure per round)"):
        if "Z_pca" in results:
            run_round_analysis(results["Z_pca"], metadata)
        else:
            print("  [skip] requires PCA — set 'pca': True in PIPELINE")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    figures = sorted(FIGURES_DIR.glob("*.png"))
    print(f"\n{'='*60}")
    print(f"Done.  {len(figures)} figures saved to: {FIGURES_DIR}/")
    for p in figures:
        print(f"  {p.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
