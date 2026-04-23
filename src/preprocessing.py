import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import (
    METADATA_COLS, PHASE_COL, PHASE_LABELS,
    COHORT_COL, COHORT_MAP, SUBJECT_COL,
    WITHIN_SUBJECT_NORMALIZE,
)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Split the DataFrame into a scaled feature matrix and a metadata DataFrame.

    Returns
    -------
    X_scaled      : (n_samples, n_features) standardised NumPy array
    feature_names : list of column names that became features
    metadata      : DataFrame with metadata + derived label columns
    """
    meta_present = [c for c in METADATA_COLS if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if c not in meta_present and pd.api.types.is_numeric_dtype(df[c])
    ]

    valid = df[feature_cols].notna().all(axis=1)
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with missing feature values")

    df_clean = df[valid].copy()
    X = df_clean[feature_cols].values.astype(float)
    metadata = df_clean[meta_present].copy()

    # Human-readable phase label
    if PHASE_COL in metadata.columns:
        metadata["phase_label"] = (
            metadata[PHASE_COL].astype(str).map(lambda v: PHASE_LABELS.get(v, v))
        )

    # Collapsed cohort (D1_3 … D1_6 → D13)
    if COHORT_COL in metadata.columns and COHORT_MAP:
        metadata["cohort_group"] = (
            metadata[COHORT_COL].astype(str).map(lambda v: COHORT_MAP.get(v, v))
        )

    # Global z-score (removes feature scale differences)
    X_scaled = StandardScaler().fit_transform(X)

    # Within-subject z-score (removes individual physiological baselines)
    if WITHIN_SUBJECT_NORMALIZE and SUBJECT_COL in metadata.columns:
        X_scaled = _within_subject_normalize(X_scaled, metadata[SUBJECT_COL].values)
        print("  Applied within-subject z-scoring")

    print(f"  Samples : {X_scaled.shape[0]}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature columns: {feature_cols}")

    return X_scaled, feature_cols, metadata


def _within_subject_normalize(X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """Z-score each feature within each subject independently."""
    X_out = X.copy()
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        if mask.sum() > 1:
            X_out[mask] = StandardScaler().fit_transform(X[mask])
    return X_out
