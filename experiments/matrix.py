"""
Full experiment matrix: datasets x drift scenarios x models x seeds.

For every cell, this fits the monitor, trains the classifier, generates a drifted stream, runs the correlation experiment (runner.py), 
and records the Spearman/Pearson correlation of each CC signal (and the KS/KL baselines) against each fairness metric. Correlations are 
averaged across seeds, since a single seed's 20-batch correlation is noisy; the seed-averaged value is the reportable result.

Scenario library (19 scenarios):
  - no_drift                          (control)
  - gradual_group   mag in {0.5,1,2}   (3)
  - gradual_global  mag in {0.5,1,2}   (3)
  - abrupt_group    mag in {0.5,1,2}   (3)
  - abrupt_global   mag in {0.5,1,2}   (3)
  - variance_group  mag in {1.5,2,2.5} (3)   [variance is a scale multiplier]
  - variance_global mag in {1.5,2,2.5} (3)

Output: one row per (dataset, scenario, model, signal, metric) with the seed-averaged Spearman and Pearson correlations (and their std 
across seeds), written to a CSV for analysis and plotting.

Note on cost: 3 datasets x 19 scenarios x 2 models x N seeds calls to the runner. The full 20-seed run if local is extremely slow.
    - The slowdown is largely due to the O(n^2) naive KDE runtime on larger datasets like Credit.
    - With 20 seeds, this is 3 x 19 x 2 x 20 = 2,280 calls to the runner.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from monitor.core import FairnessDriftMonitor
from experiments.models import train_model
from experiments.runner import run_single_experiment
from drift.config import DriftSpec, DriftType, DriftPattern
from drift.stream import generate_stream

FIELDNAMES = ["dataset", "scenario", "model", "signal", "metric", "spearman_mean", "spearman_std", "pearson_mean", "pearson_std",
              "n_seeds_valid"]

# scenario library:

# the 19 drift scenarios (affected_group=0 targets the minority for the group-specific scenarios):
def build_drift_specs(affected_group: int = 0) -> List[Tuple[DriftSpec, str]]:

    specs: List[Tuple[DriftSpec, str]] = []

    # no-drift control:
    specs.append((
        DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.ABRUPT, onset_batch=5, magnitude=0.0),
        "no_drift",
    ))

    # gradual group drift (the fairness-critical, localizable case):
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GROUP_COVARIATE, DriftPattern.GRADUAL, onset_batch=3, magnitude=mag, affected_group=affected_group),
            f"gradual_group_mag{mag}",
        ))

    # gradual global drift:
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.GRADUAL, onset_batch=3, magnitude=mag),
            f"gradual_global_mag{mag}",
        ))

    # abrupt group drift:
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GROUP_COVARIATE, DriftPattern.ABRUPT, onset_batch=5, magnitude=mag, affected_group=affected_group),
            f"abrupt_group_mag{mag}",
        ))

    # abrupt global drift:
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.ABRUPT, onset_batch=5, magnitude=mag),
            f"abrupt_global_mag{mag}",
        ))

    # variance (spread) group drift; magnitude is a scale multiplier (>1):
    for mag in (1.5, 2.0, 2.5):
        specs.append((
            DriftSpec(DriftType.GROUP_COVARIATE, DriftPattern.VARIANCE, onset_batch=3, magnitude=mag, affected_group=affected_group),
            f"variance_group_mag{mag}",
        ))

    # variance global drift:
    for mag in (1.5, 2.0, 2.5):
        specs.append((
            DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.VARIANCE, onset_batch=3, magnitude=mag), 
            f"variance_global_mag{mag}",
        ))

    return specs


# matrix runner:

# run the full matrix and write seed-averaged correlations to CSV 
# [For each (dataset, scenario, model): run one experiment per seed, then average each (signal, metric) correlation across seeds]:
def run_matrix(
    loaders: Dict[str, callable],
    seeds_by_dataset: Dict[str, List[int]],   # each dataset uses its own seeds
    models: Tuple[str, ...] = ("lr", "xgb"),
    n_batches: int = 20,
    out_path: str = "experiments/results/matrix_results.csv",
    drift_continuous_only: bool = False,
):
    specs = build_drift_specs()
 
    # start the CSV fresh with just the header; each dataset appends its rows:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
 
    total_rows = 0
    for ds_key, loader in loaders.items():
        seeds = seeds_by_dataset[ds_key]
        ds_rows = []
 
        for spec, drift_label in specs:
            for model_name in models:
                acc = defaultdict(lambda: {"spearman": [], "pearson": []})
 
                for seed in seeds:
                    bundle = loader(seed=seed)
                    monitor = FairnessDriftMonitor.from_bundle(bundle)
 
                    Xtr = np.column_stack([bundle.X_train, bundle.protected_train])
                    model, threshold = train_model(model_name, Xtr,
                                                   bundle.y_train, seed=seed)
 
                    spec_seed = _apply_continuous_only(spec, bundle, drift_continuous_only)
                    spec_seed.random_state = seed
 
                    stream = generate_stream(bundle, spec_seed,
                                             n_batches=n_batches, random_state=seed)
 
                    result = run_single_experiment(
                        bundle=bundle, monitor=monitor, model=model,
                        threshold=threshold, stream=stream,
                        drift_label=drift_label, model_name=model_name,
                    )
 
                    for signal, by_metric in result.correlations.items():
                        for metric, corr in by_metric.items():
                            key = (signal, metric)
                            acc[key]["spearman"].append(corr["spearman"])
                            acc[key]["pearson"].append(corr["pearson"])
 
                for (signal, metric), vals in acc.items():
                    sp = np.array(vals["spearman"], dtype=float)
                    pe = np.array(vals["pearson"], dtype=float)
                    ds_rows.append({
                        "dataset": ds_key, "scenario": drift_label,
                        "model": model_name, "signal": signal, "metric": metric,
                        "spearman_mean": _nanmean(sp), "spearman_std": _nanstd(sp),
                        "pearson_mean": _nanmean(pe), "pearson_std": _nanstd(pe),
                        "n_seeds_valid": int(np.sum(~np.isnan(sp))),
                    })
                print(f"[done] {ds_key} | {drift_label} | {model_name}")
 
        # incremental save: append this dataset's rows as soon as it finishes,
        # so an interruption later keeps the datasets already completed.
        _append_csv(ds_rows, out_path)
        total_rows += len(ds_rows)
        print(f"[SAVED] {ds_key}: {len(ds_rows)} rows appended to {out_path}")
 
    print(f"\nDone. Wrote {total_rows} rows total to {out_path}")


# helpers:

# return a copy of the spec, optionally restricting drift to continuous feature columns (so the monitor's input subset is what drifts):
def _apply_continuous_only(spec, bundle, drift_continuous_only):
    import copy
    s = copy.copy(spec)
    if drift_continuous_only:
        s.feature_indices = list(bundle.continuous_indices)
    return s


def _nanmean(a):
    a = a[~np.isnan(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def _nanstd(a):
    a = a[~np.isnan(a)]
    return float(np.std(a)) if len(a) else float("nan")


def _append_csv(rows, out_path):
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(rows)


if __name__ == "__main__":
    from data.loaders.load_lsac import load_lsac, CONFAIR_SEEDS as LSAC_SEEDS
    from data.loaders.load_meps import load_meps, CONFAIR_SEEDS as MEPS_SEEDS
    from data.loaders.load_credit import load_credit, CONFAIR_SEEDS as CREDIT_SEEDS
 
    # Credit LAST (slowest), so an interrupted overnight run keeps lsac + meps:
    loaders = {"lsac": load_lsac, "meps": load_meps, "credit": load_credit}
    seeds_by_dataset = {
        "lsac": LSAC_SEEDS,
        "meps": MEPS_SEEDS,
        "credit": CREDIT_SEEDS,
    }
 
    run_matrix(
        loaders=loaders,
        seeds_by_dataset=seeds_by_dataset,
        models=("lr", "xgb"),
        n_batches=20,
        out_path="experiments/results/matrix_results_all_scenarios.csv",
        drift_continuous_only=False,
    )
    