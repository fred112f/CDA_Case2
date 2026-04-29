# CDA Case II: The Representation Challenge

## Overview

This project analyzes physiological biosignal data from the **EmoPairCompete dataset** to evaluate how well feature representations encode stress responses during a competitive puzzle task. We apply multiple dimensionality reduction, statistical, and machine learning techniques to understand:

1. **Do physiological features separate by task phase?** (unsupervised)
2. **What features most strongly discriminate phases?** (feature importance)
3. **How do individuals respond to stress?** (within-subject deltas)
4. **Do physiological responses vary across repeated attempts?** (round effects)
5. **How tightly coupled are different physiological modalities?** (multimodal analysis)

---

## Data

**Source**: EmoPairCompete biosignal study
- **Participants**: 26 subjects
- **Phases per round**: 3 (pre-puzzle rest → puzzle stress → post-puzzle recovery)
- **Rounds per participant**: 4
- **Total observations**: 311 samples (1 dropped due to missing values)
- **Physiological modalities**: Heart Rate (HR), Temperature (TEMP), Electrodermal Activity (EDA)
- **Total features**: 51 (time-domain statistics and morphology metrics across the 3 sensors)

**Metadata**: Round, Phase, Subject ID, Cohort, Role (Puzzler vs Instructor), Self-report emotions (PANAS-SF, frustration VAS)

---

## Quick Start

### Installation

```bash
# Install dependencies using uv
uv sync

# Or run directly
uv run python main.py
```

### Configuration

Edit `src/config.py` to:
- Change the data file path (`HR_DATA_FILE`)
- Toggle individual pipeline steps on/off via the `PIPELINE` dictionary
- Adjust method hyperparameters (PCA components, t-SNE perplexity, etc.)

### Run the Full Pipeline

```bash
uv run python main.py
```

All figures are saved to `outputs/figures/`.

---

## Pipeline: 12 Analysis Steps

### **1. PCA Analysis**
- Applies Principal Component Analysis to the 51-dimensional feature space
- Outputs: scree plot, 2D/3D scatter projections (colored by phase/cohort/role), biplot with top feature loadings, feature correlation heatmap
- Key metric: silhouette score on phase labels (measures phase clustering quality)
- **Finding**: PC1 ≈ EDA activity, PC2 ≈ HR, PC3 ≈ Temperature. Phase silhouette = 0.009 (weak clustering)

### **2. t-SNE**
- Nonlinear manifold learning for visualization
- Outputs: 2D scatter projection colored by phase
- Key metric: silhouette score
- **Finding**: No improvement over PCA (silhouette = −0.013). Phase structure not captured by nonlinear embeddings.

### **3. UMAP**
- Uniform Manifold Approximation and Projection
- Outputs: 2D scatter projection
- Key metric: silhouette score
- **Finding**: Similar to t-SNE (silhouette = −0.010). Phases remain overlapped.

### **4. CCA (Canonical Correlation Analysis)**
- Measures how tightly HR+TEMP covary with EDA
- Outputs: canonical variates scatter, loading heatmap
- Key metric: canonical correlations
- **Finding**: r = 0.831 and 0.621 — strong coupling. Physiological modalities are coordinated.

### **5. Questionnaire Correlation**
- Pearson correlation between PC scores and self-reported emotions (PANAS-SF items + frustration)
- Outputs: heatmap with significance markers
- **Finding**: PC2 most predictive of emotional state (r = +0.25 with alertness, +0.30 with activity)

### **6. Per-Subject Trajectories**
- Visualizes how each subject moves through PC space across phases
- Outputs: individual trajectories (26 lines with arrows), mean trajectory ± SEM
- **Finding**: Clear directional flow from phase 1 → 2 → 3 despite overall weak clustering

### **7. LDA Comparison** (Supervised Upper Bound)
- Linear Discriminant Analysis using phase labels
- Outputs: LD projection, 5-fold CV accuracy
- Key metric: supervised silhouette + cross-validated accuracy
- **Finding**: LDA silhouette = 0.197, accuracy = 62.4% (vs 33% chance). Signal *is* there, distributed across many dimensions.

### **8. Permutation Test**
- Tests whether observed silhouette is above chance by comparing to 1000 random label shuffles
- Outputs: null distribution histogram with observed score and 95th percentile
- **Finding**: p ≈ 0.0000 — even the tiny observed silhouette (0.009) is statistically significant.

### **9. Role Analysis** (Puzzler vs Instructor)
- Mann-Whitney U test on PC scores between the two roles
- Outputs: scatter (role as marker shape, phase as color), boxplots by role
- **Finding**: No significant PC differences by role (all p > 0.7)

### **10. Feature Importance** ⭐ (ANOVA F-test)
- One-way ANOVA for each feature across phase labels
- Outputs: ranked bar chart (colored by sensor type), PCA re-run on top-20 features only
- **Finding**: `EDA_TD_P_Peaks` is dominant discriminator (F=85, ~4× above next). Selected-feature PCA silhouette = 0.028 (3× improvement). Noise features were diluting signal.

### **11. Delta Analysis** ⭐⭐ (Within-Subject Phase Changes)
- Computes `(phase2 − phase1)` and `(phase3 − phase1)` per subject×round
- Removes individual baselines entirely — most principled approach for within-subject design
- Outputs: mean delta bar chart ± SEM, PCA on delta vectors, delta PCA colored by role
- **Finding**: **18 features show significant stress responses** (Bonferroni-corrected):
  - EDA phasic: Peaks +1.2 σ, Median +1.0 σ, AUC +0.9 σ
  - TEMP & HR: +0.8, +0.6 σ
  - Only 3 features remain elevated during recovery
  - **Conclusion**: Physiology absolutely responds to stress. Weak PCA silhouette wasn't absence of signal — it was signal masked by between-subject noise.

### **12. Round Analysis**
- Computes silhouette score separately for each of 4 repeated rounds
- Outputs: silhouette bar chart, faceted PCA scatter (one panel per round)
- **Finding**: Clear habituation: Round 1 (sil=0.072) → Round 4 (sil=−0.017). Stress response fades with repetition.

---

## Key Findings

| Question | Answer |
|---|---|
| **Do phases cluster in PCA?** | Weakly (silhouette = 0.009), but signal is real (permutation p ≈ 0) |
| **What features discriminate phases?** | EDA phasic metrics dominate (F=85); HR and TEMP secondary |
| **Do individuals respond to stress?** | **Yes, clearly** — deltas show +0.6 to +1.2 σ changes (p < 0.0001) |
| **Is the signal consistent?** | Yes across subjects, but **habituates** from round 1 to 4 |
| **Are physiological systems linked?** | Yes — HR/TEMP and EDA show strong coupling (r=0.83) |
| **Can we predict phase from physiology alone?** | LDA: 62.4% (vs 33% chance) — better than random, but room for error |

---

## Project Structure

```
CDA_Case2/
├── main.py                        # Pipeline orchestrator
├── src/
│   ├── config.py                  # All tunable parameters & PIPELINE toggles
│   ├── loader.py                  # Data loading & validation
│   ├── preprocessing.py           # Z-scoring, within-subject normalization
│   ├── pca_analysis.py            # PCA, scree, biplot, correlations
│   ├── manifold_analysis.py       # t-SNE, UMAP
│   ├── cca_analysis.py            # Canonical Correlation Analysis
│   ├── questionnaire_analysis.py  # PC-emotion correlations
│   ├── trajectory_analysis.py     # Per-subject trajectories
│   ├── lda_analysis.py            # LDA supervised baseline
│   ├── permutation_test.py        # Null distribution test
│   ├── role_analysis.py           # Puzzler vs Instructor comparison
│   ├── feature_importance.py      # ANOVA F-test per feature
│   ├── delta_analysis.py          # Within-subject phase deltas
│   └── round_analysis.py          # Round-by-round silhouette & PCA
├── data/
│   └── HR_data.csv                # Raw biosignal data (not in repo — add locally)
├── outputs/
│   └── figures/                   # All generated plots (.png)
├── pyproject.toml                 # Project metadata & dependencies
└── README.md                      # This file
```

---

## Preprocessing

1. **Loading**: Read CSV, validate metadata columns
2. **Feature extraction**: Identify numeric columns not in metadata list (51 features identified)
3. **Cleaning**: Drop rows with missing feature values (1 row dropped)
4. **Global z-scoring**: StandardScaler across all 311 samples
5. **Within-subject z-scoring**: (Optional, enabled by default) Re-normalize each subject independently
   - Removes inter-individual physiological baseline differences
   - Allows phase effects to emerge from person-specific variations

---

## Configuration

Edit `src/config.py` to customize:

```python
# Data file
HR_DATA_FILE = Path("data") / "HR_data.csv"

# Toggle any step on/off
PIPELINE = {
    "pca": True,
    "tsne": True,
    "umap": True,
    "cca": True,
    "questionnaire_correlation": True,
    "subject_trajectories": True,
    "lda_comparison": True,
    "permutation_test": True,
    "role_analysis": True,
    "feature_importance": True,      # NEW
    "delta_analysis": True,           # NEW
    "round_analysis": True,           # NEW
}

# Method parameters
PCA_N_COMPONENTS = 10
TSNE_PERPLEXITY = 30
FEATURE_SELECT_TOP_N = 20
```

---

## Interpretation Guide

### Why is PCA silhouette so low (0.009)?

The phase effect *is* real (permutation p≈0, LDA 62.4%), but it's **small relative to between-subject noise**. 

**Analogy**: Trying to hear a whisper in a crowded stadium. The whisper is real, but drowned out by crowd noise.

**Solution**: Delta analysis removes between-subject differences by comparing each person to themselves, revealing clear responses (±1 σ).

### What does the round effect mean?

Participants **habituate** — their stress response is strongest in the first attempt (silhouette=0.072) and fades by the 4th (silhouette=−0.017). This is a known physiological phenomenon and important for study design.

### Why is CCA so strong (0.83) but phase separation weak?

**CCA** measures how well two groups of features covary with *each other*. HR, TEMP, and EDA move together as a coordinated stress response system.

**Phase separation** measures whether those response levels differ cleanly between phases when averaged across people. With high between-subject variance, even a coordinated system shows weak clustering.

---

## Output Figures

| File | What It Shows |
|---|---|
| `feature_correlations.png` | 51×51 heatmap of feature correlations (grouped by sensor) |
| `pca_scree.png` | Variance explained per component |
| `pca_projections.png` | 2D/3D PCA colored by phase/cohort/role/subject |
| `pca_biplot.png` | PC1 vs PC2 with top feature loadings as arrows |
| `pca_selected_features.png` | PCA re-run on top-20 ANOVA features |
| `tsne_projections.png` | 2D t-SNE embedding |
| `umap_projections.png` | 2D UMAP embedding |
| `cca_scatter.png` | Canonical variates scatter |
| `cca_loadings.png` | Feature loadings heatmap |
| `questionnaire_pc_correlation.png` | PC-emotion correlation heatmap |
| `trajectories_individual.png` | All 26 subject trajectories |
| `trajectories_mean.png` | Mean trajectory ± SEM |
| `lda_projection.png` | LDA space with phase coloring |
| `permutation_test.png` | Null distribution vs observed silhouette |
| `role_pca_scatter.png` | PCA with role as marker shape |
| `role_boxplots.png` | PC scores by role and phase |
| `feature_importance.png` | ANOVA F-statistics ranked |
| `delta_mean_profile.png` | Stress/recovery response by feature |
| `delta_pca.png` | PCA on phase deltas |
| `delta_pca_role.png` | Delta PCA colored by role |
| `round_silhouette.png` | Silhouette score per round |
| `round_pca_facets.png` | 4-panel PCA (one per round) |

---

## Dependencies

- **pandas** ≥ 2.0 — data manipulation
- **numpy** ≥ 1.26 — numerical operations
- **scikit-learn** ≥ 1.4 — ML algorithms (PCA, LDA, t-SNE, CCA, silhouette, ANOVA)
- **umap-learn** ≥ 0.5 — UMAP embedding
- **matplotlib** ≥ 3.8 — plotting
- **seaborn** ≥ 0.13 — statistical visualization
- **scipy** ≥ 1.12 — statistics (Mann-Whitney U, t-test)

All specified in `pyproject.toml` for `uv`.

---

## How to Cite

If you use this analysis pipeline, cite the DTU 02582 CDA Case II assignment and the EmoPairCompete dataset.

---

## Notes

- **Data confidentiality**: The `data/HR_data.csv` file is not included in the repository. Place your copy in the `data/` directory.
- **Random seeds**: Set to 42 for reproducibility (PCA, t-SNE, UMAP, LDA, permutation test).
- **Within-subject normalization**: Enabled by default. Disable via `WITHIN_SUBJECT_NORMALIZE = False` in config.py to compare with global z-scoring only.
- **Feature selection**: Delta analysis uses all 51 features; feature importance analysis identifies which are most discriminative.

---

## Contact & Questions

For questions about the pipeline, analysis methods, or interpretation, refer to the inline documentation in each `src/*.py` module.
