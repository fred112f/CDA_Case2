import pandas as pd
from pathlib import Path
from src.config import METADATA_COLS, PHASE_COL, COHORT_COL, ROUND_COL, SUBJECT_COL


def load_hr_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(
            f"\nData file not found: {filepath}\n"
            f"Place HR_data_2.csv (or HR_data.csv) in the '{filepath.parent}/' directory.\n"
            f"Then update HR_DATA_FILE in src/config.py if needed."
        )

    df = pd.read_csv(filepath)
    print(f"  Loaded: {filepath.name}  ({df.shape[0]} rows × {df.shape[1]} columns)")
    print(f"  All columns: {list(df.columns)}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in METADATA_COLS if c not in df.columns]
    if missing:
        print(
            f"\n  [!] Expected metadata columns not found in CSV: {missing}\n"
            f"  Update METADATA_COLS / *_COL variables in src/config.py\n"
            f"  to match the actual column names shown above."
        )
    else:
        for col in [PHASE_COL, COHORT_COL, ROUND_COL, SUBJECT_COL]:
            if col in df.columns:
                print(f"  {col}: {sorted(df[col].dropna().unique().tolist())}")
