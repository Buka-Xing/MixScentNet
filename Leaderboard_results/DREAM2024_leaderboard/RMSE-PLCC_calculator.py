"""
compute_metrics.py
────────────────────────────────────────────────────────────────────────────────
Compute RMSE and PLCC between the 4th-column values of two CSV files.

Inputs:
  • FILE_A / FILE_B : two CSV files
      Columns 1-3 : index keys (Dataset, Mixture_1, Mixture_2) for row alignment
      Column  4   : numeric values to compare (predictions vs. ground truth)

Matching rule:
  Rows are aligned using the first three columns as a composite key.
  Only rows present in both files are used for metric computation.
  Symmetric matching is supported: (Dataset, A, B) and (Dataset, B, A)
  are treated as the same pair.

Metrics:
  • RMSE  — Root Mean Square Error
  • PLCC  — Pearson Linear Correlation Coefficient

Dependencies: pip install pandas scipy numpy
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════════

FILE_A = "Leaderboard_set_TrueValue.csv"              # ← File A (e.g. ground truth)
FILE_B = "MixScentNet_leaderboard_predictions.csv"    # ← File B (e.g. predictions)

# Number of leading key columns; the next column (index KEY_COL_COUNT) is the value column
KEY_COL_COUNT = 3

# Enable symmetric matching: treat (Dataset, A, B) and (Dataset, B, A) as the same pair
SYMMETRIC_MATCH = True

# ════════════════════════════════════════════════════════════════════════════════


def canonical_key(d, m1, m2) -> tuple:
    """Symmetric normalized key: sort the mixture pair to eliminate ordering differences."""
    return (str(d).strip(), *sorted([str(m1).strip(), str(m2).strip()]))


def exact_key(d, m1, m2) -> tuple:
    """Exact key: preserve the original column order."""
    return (str(d).strip(), str(m1).strip(), str(m2).strip())


def load_and_index(filepath: str, key_count: int, symmetric: bool) -> tuple[dict, str]:
    """
    Read a CSV file, build an index keyed by the first key_count columns,
    and return the value column (column index key_count) as a lookup dict.
    Uses canonical (symmetric) keys when symmetric=True.
    """
    df = pd.read_csv(filepath, dtype=str)
    cols = df.columns.tolist()

    if len(cols) < key_count + 1:
        raise ValueError(
            f"{filepath} requires at least {key_count + 1} columns, "
            f"but only {len(cols)} were found."
        )

    key_cols = cols[:key_count]
    val_col  = cols[key_count]          # value column (4th column, index 3)

    print(f"  File       : {filepath}")
    print(f"  Key cols   : {key_cols}")
    print(f"  Value col  : [{val_col}] (column {key_count + 1})")
    print(f"  Total rows : {len(df)}")

    keyfunc = canonical_key if symmetric else exact_key
    index   = {}

    for _, row in df.iterrows():
        k   = keyfunc(*[row[c] for c in key_cols[:3]])   # build key from first 3 columns
        val = row[val_col]
        try:
            index[k] = float(val)
        except (ValueError, TypeError):
            index[k] = float("nan")     # unparseable values stored as NaN

    return index, val_col


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plcc(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (pearson_r, p_value)."""
    r, p = stats.pearsonr(a, b)
    return float(r), float(p)


def main():
    for f in (FILE_A, FILE_B):
        if not Path(f).exists():
            raise FileNotFoundError(f"File not found: {f}")

    print(f"{'─' * 60}")
    print("File A")
    idx_a, val_col_a = load_and_index(FILE_A, KEY_COL_COUNT, SYMMETRIC_MATCH)

    print(f"\nFile B")
    idx_b, val_col_b = load_and_index(FILE_B, KEY_COL_COUNT, SYMMETRIC_MATCH)

    # ── Align rows: keep only keys present in both files ─────────────────────
    keys_a   = set(idx_a.keys())
    keys_b   = set(idx_b.keys())
    common   = keys_a & keys_b
    only_a   = keys_a - keys_b
    only_b   = keys_b - keys_a

    print(f"\n{'─' * 60}")
    print(f"  File A total keys      : {len(keys_a)}")
    print(f"  File B total keys      : {len(keys_b)}")
    print(f"  Common keys (used)     : {len(common)}")
    print(f"  Only in File A         : {len(only_a)}  (ignored)")
    print(f"  Only in File B         : {len(only_b)}  (ignored)")

    if not common:
        print("\n❌ No common rows found. Please verify the key columns match between files.")
        return

    # ── Extract aligned values, drop NaN pairs ────────────────────────────────
    vals_a, vals_b = [], []
    nan_count = 0

    for k in common:
        va, vb = idx_a[k], idx_b[k]
        if np.isnan(va) or np.isnan(vb):
            nan_count += 1
        else:
            vals_a.append(va)
            vals_b.append(vb)

    arr_a = np.array(vals_a)
    arr_b = np.array(vals_b)
    n     = len(arr_a)

    if nan_count:
        print(f"  Rows skipped (NaN)     : {nan_count}")
    print(f"  Rows used for metrics  : {n}")

    if n < 2:
        print("\n❌ Fewer than 2 valid rows — cannot compute metrics.")
        return

    # ── Compute metrics ───────────────────────────────────────────────────────
    r_rmse          = rmse(arr_a, arr_b)
    r_plcc, p_value = plcc(arr_a, arr_b)

    print(f"\n{'═' * 60}")
    print(f"  Metrics  (File A: {val_col_a}  vs  File B: {val_col_b})")
    print(f"{'─' * 60}")
    print(f"  RMSE  : {r_rmse:.6f}")
    print(f"  PLCC  : {r_plcc:.6f}   (p-value: {p_value:.4e})")
    print(f"{'═' * 60}")

    # ── Descriptive statistics ────────────────────────────────────────────────
    print(f"\n  Descriptive statistics  ({n} rows)")
    print(f"  {'':12} {'File A':>12} {'File B':>12}")
    print(f"  {'Mean':12} {arr_a.mean():>12.4f} {arr_b.mean():>12.4f}")
    print(f"  {'Std Dev':12} {arr_a.std():>12.4f} {arr_b.std():>12.4f}")
    print(f"  {'Min':12} {arr_a.min():>12.4f} {arr_b.min():>12.4f}")
    print(f"  {'Max':12} {arr_a.max():>12.4f} {arr_b.max():>12.4f}")


if __name__ == "__main__":
    main()