"""
Baseline validation against ConFair's NO-INTERVENTION (ORIG) results.

Objective 2: validate the data + model + metric pipeline by reproducing ConFair's (Yang & Meliou, 2024) NO-INTERVENTION baseline 
fairness/utility numbers, averaged over their 20 evaluation seeds.

For each seed, this:
  1. re-splits the dataset with that seed (ConFair's permutation method)
  2. trains LR and XGBoost on ALL features WITH the protected attribute appended as a feature (matching ConFair's sensi_col_in_training=True)
  3. predicts on the test set
  4. computes DI, AOD, BalAcc with ConFair's exact formulas (fairness.py)
Then averages each metric across the 20 seeds and prints alongside ConFair's
reported targets.

Note on expected agreement:
  - LR should land close to ConFair's LR numbers.
  - XGBoost may differ more: ConFair grid-searches XGB on a validation set, whereas this project uses a fixed in-grid config and no 
    validation set (documented in bundle.py / models.py). Goal is "same ballpark", not exact.
  - Compare RAW-to-RAW (this prints raw DI/AOD/BalAcc, matching ConFair's `value` column, NOT the DI*/AOD* transformed variants).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.models import train_model, predict
from experiments.fairness import compute_metrics


# ConFair NO-INTERVENTION (ORIG) targets, averaged over their 20 seeds, computed from confair-eval.csv (group='all', method='ORIG') by 
# verify_confair_val_targets.py:
CONFAIR_TARGETS = {
    "lsac":   {"lr": dict(DI=0.193, AOD=-0.521, BalAcc=0.661),
               "xgb": dict(DI=0.353, AOD=-0.308, BalAcc=0.756)},
    "meps":   {"lr": dict(DI=0.349, AOD=-0.263, BalAcc=0.752),
               "xgb": dict(DI=0.420, AOD=-0.200, BalAcc=0.757)},
    "credit": {"lr": dict(DI=39.54, AOD=0.781, BalAcc=0.605),
               "xgb": dict(DI=5.23,  AOD=0.246, BalAcc=0.567)},  
               # Note: Credit DI mean is unstable across seeds (range 1.5-123); see confair_stats.csv
}


# run the NO-INTERVENTION baseline for one dataset across all seeds:
def run_baseline(loader,  # a loader function (e.g. load_lsac) accepting a `seed` kwarg and returning a DatasetBundle
                 seeds,  # list of seeds (the dataset's CONFAIR_SEEDS)
                 dataset_key: str  # 'lsac', 'meps', or 'credit' (for printing targets)
                ):

    results = {"lr": {"DI": [], "AOD": [], "BalAcc": []},
               "xgb": {"DI": [], "AOD": [], "BalAcc": []}}

    for seed in seeds:
        bundle = loader(seed=seed)

        # train on ALL features WITH protected attribute appended (ConFair setup):
        Xtr = np.column_stack([bundle.X_train, bundle.protected_train])
        Xte = np.column_stack([bundle.X_test, bundle.protected_test])

        for model_name in ("lr", "xgb"):
            model, threshold = train_model(model_name, Xtr, bundle.y_train, seed=seed)
            y_pred = predict(model, Xte, threshold=threshold)
            m = compute_metrics(bundle.y_test, y_pred, bundle.protected_test)
            results[model_name]["DI"].append(m["DI"])
            results[model_name]["AOD"].append(m["AOD"])
            results[model_name]["BalAcc"].append(m["BalAcc"])

    # average, print vs targets, and collect rows for CSV:
    print(f"\n=== {dataset_key.upper()} NO-INTERVENTION baseline "
          f"(mean over {len(seeds)} seeds) ===")
    print(f"{'model':<5} {'metric':<8} {'mine':>10} {'ConFair':>10} {'diff':>10}")
    rows = []
    for model_name in ("lr", "xgb"):
        for metric in ("DI", "AOD", "BalAcc"):
            mine = float(np.mean(results[model_name][metric]))
            std = float(np.std(results[model_name][metric]))
            target = CONFAIR_TARGETS[dataset_key][model_name][metric]
            print(f"{model_name:<5} {metric:<8} {mine:>10.4f} "
                  f"{target:>10.4f} {mine - target:>+10.4f}")
            rows.append({
                "dataset": dataset_key,
                "model": model_name,
                "metric": metric,
                "mine_mean": round(mine, 6),
                "mine_std": round(std, 6),
                "confair": target,
                "diff": round(mine - target, 6),
                "n_seeds": len(seeds),
            })
 
    return rows


# write collected comparison rows to a CSV (mirrors ConFair's tabular eval):
def write_csv(all_rows, out_path="baseline_validation.csv"):
    import csv
    fieldnames = ["dataset", "model", "metric", "mine_mean", "mine_std", "confair", "diff", "n_seeds"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    from data.loaders.load_lsac import load_lsac, CONFAIR_SEEDS as LSAC_SEEDS
    from data.loaders.load_meps import load_meps, CONFAIR_SEEDS as MEPS_SEEDS
    from data.loaders.load_credit import load_credit, CONFAIR_SEEDS as CREDIT_SEEDS

    all_rows = []
    all_rows += run_baseline(load_lsac, LSAC_SEEDS, "lsac")
    all_rows += run_baseline(load_meps, MEPS_SEEDS, "meps")
    all_rows += run_baseline(load_credit, CREDIT_SEEDS, "credit")
    write_csv(all_rows, "experiments/confair-validation/baseline_validation.csv")