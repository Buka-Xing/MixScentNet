"""
eval_metrics.py
────────────────────────────────────────────────────────────────────────────────
Compute per-row PLCC and Cosine Distance on 51-dimensional label vectors
across two CSV files.

Inputs:
  • FILE_A / FILE_B
      Column 1    : mixture name (used for row alignment)
      Columns 2-52: 51-dimensional label values

Outputs:
  • Console : mean / std / quantile distribution across all rows
  • eval_metrics_report.csv : per-row detailed results

Dependencies: pip install pandas numpy scipy
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════════

FILE_A        = "TASK2_Leaderboard_ActualValue.csv"          # ← File A (e.g. Ground Truth)
FILE_B        = "MixScentNet_leaderboard_predictions.csv"    # ← File B (e.g. Prediction)
OUTPUT_REPORT = "eval_metrics_report.csv"                    # ← Per-row report output path

LABEL_COL_START = 1     # Start index of label columns (0-indexed); column 2 = index 1
LABEL_COL_END   = 52    # End index of label columns (0-indexed, exclusive); column 52 = index 51

# ════════════════════════════════════════════════════════════════════════════════


def load_file(filepath: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Read a CSV file, parse the name column and label columns; coerce label columns to numeric."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df   = pd.read_csv(filepath, dtype=str)
    cols = df.columns.tolist()

    if len(cols) < LABEL_COL_END:
        raise ValueError(
            f"{filepath} has {len(cols)} columns, "
            f"but at least {LABEL_COL_END} are required (columns 2-52 are label columns)."
        )

    name_col   = cols[0]
    label_cols = cols[LABEL_COL_START:LABEL_COL_END]

    df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce")

    print(f"  File        : {filepath}")
    print(f"  Name col    : [{name_col}]")
    print(f"  Label cols  : {len(label_cols)} columns  ({label_cols[0]} … {label_cols[-1]})")
    print(f"  Total rows  : {len(df)}")

    return df, name_col, label_cols


def row_plcc(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Pearson linear correlation coefficient; returns NaN for constant vectors."""
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return float("nan")
    r, _ = stats.pearsonr(vec_a, vec_b)
    return float(r)


def row_cosine_dist(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity; returns NaN for zero vectors."""
    if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
        return float("nan")
    return float(cosine_dist(vec_a, vec_b))


def print_distribution(name: str, arr: np.ndarray) -> None:
    """Print the quantile distribution of a metric array."""
    print(f"\n  {name} distribution")
    print(f"    Min: {arr.min():.4f}  "
          f"Q1: {np.percentile(arr, 25):.4f}  "
          f"Median: {np.median(arr):.4f}  "
          f"Q3: {np.percentile(arr, 75):.4f}  "
          f"Max: {arr.max():.4f}")


def main():
    # ── 1. Load files ────────────────────────────────────────────────────────
    print(f"{'─' * 65}")
    print("File A")
    df_a, name_col_a, label_cols_a = load_file(FILE_A)

    print(f"\nFile B")
    df_b, name_col_b, label_cols_b = load_file(FILE_B)

    # ── 2. Build name-keyed index ────────────────────────────────────────────
    idx_a = {str(row[name_col_a]).strip(): row[label_cols_a].values.astype(float)
             for _, row in df_a.iterrows()}
    idx_b = {str(row[name_col_b]).strip(): row[label_cols_b].values.astype(float)
             for _, row in df_b.iterrows()}

    common = sorted(set(idx_a) & set(idx_b))
    only_a = set(idx_a) - set(idx_b)
    only_b = set(idx_b) - set(idx_a)

    print(f"\n{'─' * 65}")
    print(f"  File A mixtures              : {len(idx_a)}")
    print(f"  File B mixtures              : {len(idx_b)}")
    print(f"  Common mixtures (used)       : {len(common)}")
    print(f"  Only in File A               : {len(only_a)}  (ignored)")
    print(f"  Only in File B               : {len(only_b)}  (ignored)")

    if not common:
        print("\n❌ No common mixtures found. Please verify the name column contents match.")
        return

    # ── 3. Compute per-row metrics ───────────────────────────────────────────
    records    = []
    plcc_vals  = []
    cos_vals   = []
    nan_plcc   = 0
    nan_cos    = 0

    for name in common:
        vec_a = idx_a[name]
        vec_b = idx_b[name]

        # Keep only dimensions where neither vector is NaN
        valid = ~(np.isnan(vec_a) | np.isnan(vec_b))
        va, vb   = vec_a[valid], vec_b[valid]
        n_valid  = int(valid.sum())

        if n_valid < 2:
            p, c = float("nan"), float("nan")
        else:
            p = row_plcc(va, vb)
            c = row_cosine_dist(va, vb)

        if np.isnan(p): nan_plcc += 1
        else:           plcc_vals.append(p)

        if np.isnan(c): nan_cos += 1
        else:           cos_vals.append(c)

        records.append({
            "Mixture_Name":    name,
            "Valid_Dims":      n_valid,
            "PLCC":            round(p, 6) if not np.isnan(p) else "",
            "Cosine_Distance": round(c, 6) if not np.isnan(c) else "",
        })

    # ── 4. Aggregate statistics ──────────────────────────────────────────────
    plcc_arr = np.array(plcc_vals)
    cos_arr  = np.array(cos_vals)

    def safe_stat(arr):
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr))

    mean_plcc, std_plcc = safe_stat(plcc_arr)
    mean_cos,  std_cos  = safe_stat(cos_arr)

    print(f"\n{'═' * 65}")
    print(f"  Evaluation results  ({len(common)} mixtures)")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<22} {'Mean':>12} {'Std Dev':>12} {'NaN rows':>8}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*8}")
    print(f"  {'PLCC':<22} {mean_plcc:>12.6f} {std_plcc:>12.6f} {nan_plcc:>8}")
    print(f"  {'Cosine Distance':<22} {mean_cos:>12.6f} {std_cos:>12.6f} {nan_cos:>8}")
    print(f"{'═' * 65}")

    if len(plcc_arr) > 0:
        print_distribution("PLCC", plcc_arr)
    if len(cos_arr) > 0:
        print_distribution("Cosine Distance", cos_arr)

    # ── 5. Save per-row report ───────────────────────────────────────────────
    pd.DataFrame(records).to_csv(OUTPUT_REPORT, index=False, encoding="utf-8-sig")
    print(f"\n📁 Per-row report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()