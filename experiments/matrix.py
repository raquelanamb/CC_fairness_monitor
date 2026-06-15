"""
Full experiment matrix: datasets x drift scenarios x models x seeds.

For every cell, this fits the monitor, trains the classifier, generates a drifted stream, runs the correlation experiment (runner.py), 
and records the Spearman/Pearson correlation of each CC signal (and the KS/KL baselines) against each fairness metric. Correlations are 
averaged across seeds, since a single seed's 20-batch correlation is noisy; the seed-averaged value is the reportable result.

Scenario library (13 scenarios), carried forward from the prior design:
  - no_drift                          (control)
  - gradual_group  mag in {0.5,1,2}   (3)
  - gradual_global mag in {0.5,1,2}   (3)
  - abrupt_global  mag in {0.5,1,2}   (3)
  - variance_group mag in {1.5,2,2.5} (3)   [variance is a scale multiplier]

Output: one row per (dataset, scenario, model, signal, metric) with the seed-averaged Spearman and Pearson correlations (and their std 
across seeds), written to a CSV for analysis and plotting.

Note on cost: 3 datasets x 13 scenarios x 2 models x N seeds calls to the runner. Use a small N (e.g. 3) for a first full-matrix smoke run, 
then 20 for the reportable results. The HPC is the place for the full 20-seed run if local is slow.
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


# scenario library:

# the 13 drift scenarios (affected_group=0 targets the minority for the group-specific scenarios):
def build_drift_specs(affected_group: int = 0) -> List[Tuple[DriftSpec, str]]:

    specs: List[Tuple[DriftSpec, str]] = []

    # no-drift control:
    specs.append((
        DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.ABRUPT,
                  onset_batch=5, magnitude=0.0),
        "no_drift",
    ))

    # gradual group drift (the fairness-critical, localizable case):
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GROUP_COVARIATE, DriftPattern.GRADUAL,
                      onset_batch=3, magnitude=mag, affected_group=affected_group),
            f"gradual_group_mag{mag}",
        ))

    # gradual global drift:
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.GRADUAL,
                      onset_batch=3, magnitude=mag),
            f"gradual_global_mag{mag}",
        ))

    # abrupt global drift:
    for mag in (0.5, 1.0, 2.0):
        specs.append((
            DriftSpec(DriftType.GLOBAL_COVARIATE, DriftPattern.ABRUPT,
                      onset_batch=5, magnitude=mag),
            f"abrupt_global_mag{mag}",
        ))

    # variance (spread) group drift; magnitude is a scale multiplier (>1):
    for mag in (1.5, 2.0, 2.5):
        specs.append((
            DriftSpec(DriftType.GROUP_COVARIATE, DriftPattern.VARIANCE,
                      onset_batch=3, magnitude=mag, affected_group=affected_group),
            f"variance_group_mag{mag}",
        ))

    return specs


# matrix runner:

# run the full matrix and write seed-averaged correlations to CSV 
# [For each (dataset, scenario, model): run one experiment per seed, then average each (signal, metric) correlation across seeds]:
def run_matrix(
    loaders: Dict[str, callable],   # {dataset_key: loader_fn}
    seeds: List[int],
    models: Tuple[str, ...] = ("lr", "xgb"),
    n_batches: int = 20,
    out_path: str = "experiments/results/matrix_results.csv",
    drift_continuous_only: bool = False,
):

    specs = build_drift_specs()
    rows = []

    for ds_key, loader in loaders.items():
        for spec, drift_label in specs:
            for model_name in models:
                # accumulate per-seed correlations:
                # acc[(signal, metric)] -> {"spearman": [...], "pearson": [...]}
                acc = defaultdict(lambda: {"spearman": [], "pearson": []})

                for seed in seeds:
                    bundle = loader(seed=seed)

                    # fit monitor on training data:
                    monitor = FairnessDriftMonitor.from_bundle(bundle)

                    # train classifier WITH protected col appended (baseline setup):
                    Xtr = np.column_stack([bundle.X_train, bundle.protected_train])
                    model, threshold = train_model(model_name, Xtr,
                                                   bundle.y_train, seed=seed)

                    # optionally restrict drift to continuous features:
                    spec_seed = _apply_continuous_only(spec, bundle, drift_continuous_only)
                    spec_seed.random_state = seed

                    stream = generate_stream(bundle, spec_seed,
                                             n_batches=n_batches, random_state=seed)

                    result = run_single_experiment(
                        bundle=bundle, monitor=monitor, model=model,
                        threshold=threshold, stream=stream,
                        drift_label=drift_label, model_name=model_name,
                    )

                    # collect every (signal, metric) correlation for this seed:
                    for signal, by_metric in result.correlations.items():
                        for metric, corr in by_metric.items():
                            key = (signal, metric)
                            acc[key]["spearman"].append(corr["spearman"])
                            acc[key]["pearson"].append(corr["pearson"])

                # average across seeds (nanmean: skip undefined-correlation seeds):
                for (signal, metric), vals in acc.items():
                    sp = np.array(vals["spearman"], dtype=float)
                    pe = np.array(vals["pearson"], dtype=float)
                    rows.append({
                        "dataset": ds_key,
                        "scenario": drift_label,
                        "model": model_name,
                        "signal": signal,
                        "metric": metric,
                        "spearman_mean": _nanmean(sp),
                        "spearman_std": _nanstd(sp),
                        "pearson_mean": _nanmean(pe),
                        "pearson_std": _nanstd(pe),
                        "n_seeds_valid": int(np.sum(~np.isnan(sp))),
                    })
                print(f"[done] {ds_key} | {drift_label} | {model_name}")

    _write_csv(rows, out_path)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    return rows


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


def _write_csv(rows, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "scenario", "model", "signal", "metric", "spearman_mean", "spearman_std", "pearson_mean", "pearson_std",
                  "n_seeds_valid"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    from data.loaders.load_lsac import load_lsac, CONFAIR_SEEDS as LSAC_SEEDS
    from data.loaders.load_meps import load_meps, CONFAIR_SEEDS as MEPS_SEEDS
    from data.loaders.load_credit import load_credit, CONFAIR_SEEDS as CREDIT_SEEDS

    loaders = {
        "lsac": load_lsac,
        "meps": load_meps,
        "credit": load_credit,
    }

    # FIRST full-matrix run - use a small seed count to verify the whole matrix runs end-to-end, then switch to the full 20 seeds for 
    # reportable result:
    quick_seeds = LSAC_SEEDS[:3]   # same seeds across datasets for the quick run

    run_matrix(
        loaders=loaders,
        seeds=quick_seeds,
        models=("lr", "xgb"),
        n_batches=20,
        out_path="experiments/results/matrix_results_quick.csv",
        drift_continuous_only=False,
    )