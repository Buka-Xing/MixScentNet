"""
eval_last_col.py
────────────────────────────────────────────────────────────────────────────────
Compute PLCC and MSE between the last-column values of two CSV files.

Matching rule:
  All columns except the last are used as a composite key to align rows.
  Only rows present in both files are used for metric computation.

Outputs:
  • Console : PLCC / MSE and descriptive statistics
  • last_col_metrics_report.csv : per-row aligned results

Dependencies: pip install pandas numpy scipy
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════════

FILE_A        = "MixScentNet_predictions.csv"         # ← File A
FILE_B        = "Test_set_mixturedist.csv"            # ← File B
OUTPUT_REPORT = "last_col_metrics_report.csv"         # ← Per-row alignment report

# ════════════════════════════════════════════════════════════════════════════════


def load_file(filepath: str) -> tuple[pd.DataFrame, list[str], str]:
    """
    Read a CSV file, auto-detect key columns (all except the last)
    and the value column (the last column).
    Returns (DataFrame, key_cols, value_col).
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df   = pd.read_csv(filepath, dtype=str)
    cols = df.columns.tolist()

    if len(cols) < 1:
        raise ValueError(f"{filepath} has no columns.")

    key_cols  = cols[:-1]   # all columns except the last serve as composite key
    value_col = cols[-1]    # last column is the numeric value to compare

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    print(f"  File      : {filepath}")
    print(f"  Key cols  : {key_cols if key_cols else '(none — rows aligned by position)'}")
    print(f"  Value col : [{value_col}] (last column)")
    print(f"  Total rows: {len(df)}")

    return df, key_cols, value_col


def build_index(df: pd.DataFrame, key_cols: list[str], value_col: str) -> dict:
    """Build a {key_tuple: float} lookup dict keyed by the composite key columns."""
    index = {}
    for _, row in df.iterrows():
        key = tuple(str(row[c]).strip() for c in key_cols) if key_cols \
              else (str(_),)
        try:
            index[key] = float(row[value_col])
        except (ValueError, TypeError):
            index[key] = float("nan")
    return index


def main():
    # ── 1. Load files ────────────────────────────────────────────────────────
    print(f"{'─' * 60}")
    print("File A")
    df_a, key_cols_a, val_col_a = load_file(FILE_A)

    print(f"\nFile B")
    df_b, key_cols_b, val_col_b = load_file(FILE_B)

    # ── 2. Build index and align rows ────────────────────────────────────────
    idx_a = build_index(df_a, key_cols_a, val_col_a)
    idx_b = build_index(df_b, key_cols_b, val_col_b)

    common = sorted(set(idx_a) & set(idx_b))
    only_a = set(idx_a) - set(idx_b)
    only_b = set(idx_b) - set(idx_a)

    print(f"\n{'─' * 60}")
    print(f"  File A rows (unique keys) : {len(idx_a)}")
    print(f"  File B rows (unique keys) : {len(idx_b)}")
    print(f"  Common rows (used)        : {len(common)}")
    print(f"  Only in File A            : {len(only_a)}  (ignored)")
    print(f"  Only in File B            : {len(only_b)}  (ignored)")

    if not common:
        print("\n❌ No common rows found. Please verify the key column contents match.")
        return

    # ── 3. Extract aligned values, drop NaN pairs ────────────────────────────
    vals_a, vals_b, valid_keys = [], [], []
    nan_count = 0

    for k in common:
        va, vb = idx_a[k], idx_b[k]
        if np.isnan(va) or np.isnan(vb):
            nan_count += 1
        else:
            vals_a.append(va)
            vals_b.append(vb)
            valid_keys.append(k)

    arr_a = np.array(vals_a)
    arr_b = np.array(vals_b)
    n     = len(arr_a)

    if nan_count:
        print(f"  Rows skipped (NaN)        : {nan_count}")
    print(f"  Rows used for metrics     : {n}")

    if n < 2:
        print("\n❌ Fewer than 2 valid rows — cannot compute metrics.")
        return

    # ── 4. Compute PLCC and MSE ──────────────────────────────────────────────
    plcc, p_value = stats.pearsonr(arr_a, arr_b)
    mse           = float(np.mean((arr_a - arr_b) ** 2))
    rmse          = float(np.sqrt(mse))

    print(f"\n{'═' * 60}")
    print(f"  Metrics  [{val_col_a}] (A)  vs  [{val_col_b}] (B)")
    print(f"{'─' * 60}")
    print(f"  PLCC : {plcc:.6f}   (p-value: {p_value:.4e})")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}   (square root of MSE, same unit as the data)")
    print(f"{'═' * 60}")

    # ── 5. Descriptive statistics ────────────────────────────────────────────
    print(f"\n  Descriptive statistics  ({n} rows)")
    print(f"  {'':10} {'File A':>12} {'File B':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12}")
    for label, a_val, b_val in [
        ("Mean",    arr_a.mean(), arr_b.mean()),
        ("Std Dev", arr_a.std(),  arr_b.std()),
        ("Min",     arr_a.min(),  arr_b.min()),
        ("Max",     arr_a.max(),  arr_b.max()),
    ]:
        print(f"  {label:<10} {a_val:>12.4f} {b_val:>12.4f}")

    # ── 6. Save per-row report ───────────────────────────────────────────────
    report_rows = []
    for k, va, vb in zip(valid_keys, vals_a, vals_b):
        row = {f"key_{i}": v for i, v in enumerate(k)}
        row[f"{val_col_a}(A)"] = va
        row[f"{val_col_b}(B)"] = vb
        row["Diff(A-B)"]       = round(va - vb, 6)
        row["Squared_Error"]   = round((va - vb) ** 2, 6)
        report_rows.append(row)

    pd.DataFrame(report_rows).to_csv(OUTPUT_REPORT, index=False, encoding="utf-8-sig")
    print(f"\n📁 Per-row report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()