from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"

# Change to "HR_data.csv" if that is the file you received
HR_DATA_FILE = DATA_DIR / "HR_data.csv"

# ---------------------------------------------------------------------------
# Metadata columns — everything that is NOT a biosignal feature.
# Questionnaire responses are excluded from the feature matrix but kept
# in metadata so they can be used for coloring or CCA.
# ---------------------------------------------------------------------------
METADATA_COLS = [
    "Unnamed: 0",                                          # row index
    "Round", "Phase", "Individual", "Cohort", "Puzzler",  # experimental
    "Frustrated",                                          # 0-10 VAS
    "upset", "hostile", "alert", "ashamed", "inspired",   # I-PANAS-SF
    "nervous", "attentive", "afraid", "active", "determined",
]

PHASE_COL = "Phase"       # values: phase1 / phase2 / phase3
COHORT_COL = "Cohort"     # raw session values: D1_1 … D1_6
ROUND_COL = "Round"       # values: round_1 … round_4
SUBJECT_COL = "Individual"
PUZZLER_COL = "Puzzler"   # role: instructor vs puzzle-solver

# Collapse the 6 session labels into the 3 original cohorts for cleaner plots.
COHORT_MAP: dict[str, str] = {
    "D1_1": "D11 (Winter)",
    "D1_2": "D12 (Fall-A)",
    "D1_3": "D13 (Fall-B)",
    "D1_4": "D13 (Fall-B)",
    "D1_5": "D13 (Fall-B)",
    "D1_6": "D13 (Fall-B)",
}

# Within-subject z-scoring: removes individual physiological baselines so
# phase effects are not swamped by between-person variance.
WITHIN_SUBJECT_NORMALIZE = True

# Map raw phase values → human-readable labels used in plots.
# Add / adjust keys to match what is actually in your CSV.
PHASE_LABELS: dict[str, str] = {
    "phase1": "Pre-puzzle (Rest)",
    "phase2": "Puzzle (Stress)",
    "phase3": "Post-puzzle (Recovery)",
    "pre": "Pre-puzzle (Rest)",
    "puzzle": "Puzzle (Stress)",
    "post": "Post-puzzle (Recovery)",
    "1": "Pre-puzzle (Rest)",
    "2": "Puzzle (Stress)",
    "3": "Post-puzzle (Recovery)",
}

# ---------------------------------------------------------------------------
# Feature group prefixes — used to split features for CCA.
# Adjust if your column names use different prefixes.
# ---------------------------------------------------------------------------
HR_PREFIXES: tuple[str, ...] = ("HR",)
EDA_PREFIXES: tuple[str, ...] = ("EDA",)
TEMP_PREFIXES: tuple[str, ...] = ("TEMP",)

# ---------------------------------------------------------------------------
# Method hyperparameters
# ---------------------------------------------------------------------------
PCA_N_COMPONENTS = 10
PCA_BIPLOT_N_FEATURES = 8  # number of feature arrows shown in biplot

FEATURE_SELECT_TOP_N = 20  # top-N features selected by ANOVA F-test for phase discrimination

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
TSNE_RANDOM_STATE = 42

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

CCA_N_COMPONENTS = 2

# ---------------------------------------------------------------------------
# Plot aesthetics
# ---------------------------------------------------------------------------
PHASE_PALETTE: dict[str, str] = {
    "Pre-puzzle (Rest)": "#4C72B0",
    "Puzzle (Stress)": "#C44E52",
    "Post-puzzle (Recovery)": "#55A868",
}

FIGURE_DPI = 150
FIGURE_FORMAT = "png"

# ---------------------------------------------------------------------------
# Pipeline — set any step to False to skip it
# ---------------------------------------------------------------------------
PIPELINE: dict[str, bool] = {
    "pca": True,                      # PCA + scree + biplot + feature correlations
    "tsne": True,                     # t-SNE
    "umap": True,                     # UMAP
    "cca": True,                      # CCA (HR+TEMP vs EDA)
    "questionnaire_correlation": True, # Pearson r: PC scores vs self-report
    "subject_trajectories": True,      # Per-subject arrows through PC space
    "lda_comparison": True,            # LDA supervised upper-bound + CV accuracy
    "permutation_test": True,          # Null distribution for silhouette score
    "role_analysis": True,             # Puzzler vs Instructor comparison
    "feature_importance": True,        # ANOVA F-test per feature + selected-feature PCA
    "delta_analysis": True,            # Within-subject phase deltas (phase2-phase1, phase3-phase1)
    "round_analysis": True,            # Silhouette + PCA facets per round
}
