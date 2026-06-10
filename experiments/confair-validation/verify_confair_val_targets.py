"""
Independently reproduce ConFair's NO-INTERVENTION (ORIG) baseline statistics from their released per-seed evaluation data, for both the 
CONFAIR and DIFFAIR evaluation files. Confirms the comparison targets used in my baseline.py validation.
"""

import pandas as pd
import csv

EVAL_FILES = {
    "confair": "experiments/confair-validation/confair-eval.csv",
    "diffair": "experiments/confair-validation/diffair-eval.csv",
}

DATASETS = ["lsac", "meps16", "credit"]
MODELS = ["LR", "TR"]  # TR = their XGBoost
METRICS = ["DI", "AvgOddsDiff", "BalAcc"]


# print mean/median/std/min/max of ORIG per-seed values for each cell:
def summarize(path, source_name):
    df = pd.read_csv(path)
    orig = df[(df["method"] == "ORIG") & (df["group"] == "all")]

    print(f"\n{'='*82}")
    print(f"  {source_name}  ({path})")
    print(f"{'='*82}")
    header = (f"{'dataset':<8}{'model':<5}{'metric':<12}{'n':>4}"
              f"{'mean':>10}{'median':>10}{'std':>10}{'min':>10}{'max':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for ds in DATASETS:
        for model in MODELS:
            for metric in METRICS:
                sub = orig[(orig["data"] == ds) &
                           (orig["model"] == model) &
                           (orig["metric"] == metric)]["value"]
                if len(sub) == 0:
                    print(f"{ds:<8}{model:<5}{metric:<12}{'--':>4}  (not found)")
                    continue
                print(f"{ds:<8}{model:<5}{metric:<12}{len(sub):>4}"
                      f"{sub.mean():>10.4f}{sub.median():>10.4f}{sub.std():>10.4f}"
                      f"{sub.min():>10.4f}{sub.max():>10.4f}")
                rows.append({
                    "source": source_name,
                    "dataset": ds,
                    "model": model,
                    "metric": metric,
                    "n": len(sub),
                    "mean": round(sub.mean(), 6),
                    "median": round(sub.median(), 6),
                    "std": round(sub.std(), 6),
                    "min": round(sub.min(), 6),
                    "max": round(sub.max(), 6),
                })
    return rows

# confirm ORIG values are identical across the two eval files:
def check_files_agree():
    print(f"\n{'='*82}")
    print("  CROSS-FILE CHECK: do confair & diffair agree on ORIG?")
    print(f"{'='*82}")
    c = pd.read_csv(EVAL_FILES["confair"])
    d = pd.read_csv(EVAL_FILES["diffair"])
    cf = c[(c["method"] == "ORIG") & (c["group"] == "all")]
    df = d[(d["method"] == "ORIG") & (d["group"] == "all")]
    all_match = True
    for ds in DATASETS:
        for model in MODELS:
            for metric in METRICS:
                cv = cf[(cf["data"] == ds) & (cf["model"] == model) &
                        (cf["metric"] == metric)]["value"].mean()
                dv = df[(df["data"] == ds) & (df["model"] == model) &
                        (df["metric"] == metric)]["value"].mean()
                if abs(cv - dv) > 1e-6:
                    all_match = False
                    print(f"  MISMATCH {ds} {model} {metric}: "
                          f"confair={cv:.4f} diffair={dv:.4f}")
    print("  All ORIG means identical across both files." if all_match
          else "  Some values differ (see above).")


if __name__ == "__main__":
    all_rows = []
    for source_name, path in EVAL_FILES.items():
        all_rows += summarize(path, source_name)
    check_files_agree()

    with open("experiments/confair-validation/confair_stats.csv", "w", newline="") as f:
        fieldnames = ["source", "dataset", "model", "metric", "n",
                      "mean", "median", "std", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nWrote confair_stats.csv")