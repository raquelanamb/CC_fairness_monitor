"""
Compare two matrix.py result files from repeated runs to test same-machine reproducibility.

The unit is the cell, keyed by (dataset, scenario, model, signal, metric). Two runs of the same configuration must produce the same set of 
cells and, within an environment, the same values. This reports three things in order:
  1. Cell alignment - same cells in both runs, or cells present in only one.
  2. Value equality - per numeric column: exactly equal, equal up to tolerance, or differing, with NaN treated as a value (both NaN agree, 
     one NaN disagrees).
  3. Divergence locus - if anything differs, where it concentrates (model, dataset), since floating-point nondeterminism tends to localize 
     (e.g. XGBoost threads).
     
Verdict:
  IDENTICAL - same cells, every value bit-for-bit equal, NaN patterns match.
  DETERMINISTIC (tol) - same cells, all differences within tol; report as same-environment deterministic and name where the residual 
                        differences sit.
  DIVERGENT - missing cells or differences beyond tol.

n_seeds_valid is compared too: a cell that yielded a valid correlation in one run but not the other is a determinism flag independent of the 
float values.
"""

import sys

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)

KEY = ["dataset", "scenario", "model", "signal", "metric"]
VALUE_COLS = ["spearman_mean", "spearman_std", "pearson_mean", "pearson_std", "n_seeds_valid"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dup = df.duplicated(subset=KEY).sum()
    if dup:
        print(f"WARNING {path}: {dup} duplicate cell keys; keeping first of each")
        df = df.drop_duplicates(subset=KEY, keep="first")
    return df


def align(a: pd.DataFrame, b: pd.DataFrame):
    merged = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = merged[merged["_merge"] == "both"].copy()
    only_a = merged[merged["_merge"] == "left_only"]
    only_b = merged[merged["_merge"] == "right_only"]
    return both, only_a, only_b


def _equal_mask(a: pd.Series, b: pd.Series):
    both_nan = a.isna() & b.isna()  # both-NaN counts as agreement; exactly one NaN counts as disagreement
    one_nan = a.isna() ^ b.isna()
    eq_exact = (a == b) | both_nan
    return eq_exact.fillna(False), one_nan


def compare_values(both: pd.DataFrame, atol: float, rtol: float):
    cols = [c for c in VALUE_COLS if f"{c}_a" in both.columns and f"{c}_b" in both.columns]
    rows = []
    per_col_diff = {}
    for c in cols:
        a, b = both[f"{c}_a"], both[f"{c}_b"]
        eq_exact, one_nan = _equal_mask(a, b)
        both_nan = a.isna() & b.isna()
        close = pd.Series(np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), atol=atol, rtol=rtol, equal_nan=True),index=both.index,)
        comparable = a.notna() & b.notna()
        diff = (a - b).where(comparable)
        differing = comparable & ~close
        per_col_diff[c] = differing
        if differing.any():
            idx = diff.abs().idxmax()
            max_abs = float(diff.abs().max())
            argmax_key = " / ".join(str(both.loc[idx, k]) for k in KEY)
        else:
            max_abs, argmax_key = 0.0, ""
        rows.append({
            "column": c,
            "compared": int(comparable.sum()),
            "exact_eq": int((eq_exact & comparable).sum()),
            "both_nan": int(both_nan.sum()),
            "nan_mismatch": int(one_nan.sum()),
            "within_tol": int((close & comparable).sum()),
            "differing": int(differing.sum()),
            "max_abs_diff": max_abs,
            "mean_abs_diff": float(diff[differing].abs().mean()) if differing.any() else 0.0,
            "argmax_cell": argmax_key,
        })
    return pd.DataFrame(rows), per_col_diff


def locate(both: pd.DataFrame, per_col_diff: dict):
    any_diff = pd.Series(False, index=both.index)
    for d in per_col_diff.values():
        any_diff |= d.fillna(False)
    if not any_diff.any():
        return None
    sub = both[any_diff]
    by_model = sub.groupby("model").size().rename("differing_cells")
    by_dataset = sub.groupby("dataset").size().rename("differing_cells")
    return by_model, by_dataset


def verdict(only_a, only_b, summary: pd.DataFrame, locus) -> None:
    same_cells = len(only_a) == 0 and len(only_b) == 0
    any_nan_mismatch = summary["nan_mismatch"].sum() > 0
    any_differing = summary["differing"].sum() > 0
    all_exact = (summary["exact_eq"] == summary["compared"]).all() and not any_nan_mismatch

    print("VERDICT")
    if same_cells and all_exact:
        print("IDENTICAL: same cells, every value bit-for-bit equal. Reproducible on this machine without qualification.")
        return
    if same_cells and not any_differing and not any_nan_mismatch:
        print("DETERMINISTIC (within tolerance): same cells, all differences below tol. "
              "Report as same-environment deterministic up to floating point.")
    elif same_cells:
        print("DIVERGENT: same cells, but values differ beyond tol or NaN patterns disagree.")
    else:
        print(f"DIVERGENT: cell sets differ ({len(only_a)} only in run A, {len(only_b)} only in run B). Runs are not comparable as-is.")
    if locus is not None:
        by_model, by_dataset = locus
        print("differing cells by model:")
        print(by_model.to_string())
        print("differing cells by dataset:")
        print(by_dataset.to_string())


def main(path_a: str, path_b: str, atol: float = 0.0, rtol: float = 0.0) -> None:
    a, b = load(path_a), load(path_b)
    print(f"run A: {path_a}  ({len(a)} cells)")
    print(f"run B: {path_b}  ({len(b)} cells)")
    print(f"tolerance: atol={atol} rtol={rtol}")

    both, only_a, only_b = align(a, b)
    print(f"alignment: {len(both)} shared, {len(only_a)} only in A, {len(only_b)} only in B")
    if len(only_a) or len(only_b):
        miss = pd.concat([only_a, only_b])[KEY].head(20)
        print("examples of non-shared cells:")
        print(miss.to_string(index=False))

    summary, per_col_diff = compare_values(both, atol, rtol)
    print("per-column comparison:")
    print(summary.round(12).to_string(index=False))

    locus = locate(both, per_col_diff)
    verdict(only_a, only_b, summary, locus)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python crosscheck.py <run_a.csv> <run_b.csv> [atol] [rtol]")
        sys.exit(1)
    a_path, b_path = sys.argv[1], sys.argv[2]
    atol_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    rtol_arg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    main(a_path, b_path, atol_arg, rtol_arg)