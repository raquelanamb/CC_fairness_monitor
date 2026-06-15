"""
Smoke test: one dataset, one drift scenario, one model, end-to-end.

Purpose: verify the whole pipeline runs and produces sane output BEFORE scaling to the full experiment matrix. Specifically it checks:
  1. the monitor fits (all four subgroups, or sensibly skips small ones)
  2. baseline violations are NOT near-zero (the live sigma-bounds question: if violations are ~0 everywhere, the 4-sigma bounds are too loose 
     and we switch to percentile bounds)
  3. violations RISE as drift is injected (the signal works)
  4. correlations compute without crashing
"""

import numpy as np

from data.loaders.load_lsac import load_lsac
from monitor.core import FairnessDriftMonitor
from experiments.models import train_model
from experiments.runner import run_single_experiment
from drift.config import DriftSpec, DriftType, DriftPattern
from drift.stream import generate_stream


def main():
    print("Smoke test: LSAC, gradual group drift (minority), LR")

    # load data:
    bundle = load_lsac(seed=42)
    print(f"\nLoaded {bundle.dataset_name}: "
          f"X_train {bundle.X_train.shape}, X_test {bundle.X_test.shape}")
    print(f"continuous_indices: {bundle.continuous_indices}")

    # fit the monitor:
    monitor = FairnessDriftMonitor.from_bundle(bundle)
    print(f"\nFitted subgroups: {monitor.subgroup_names}")
    print(f"Skipped subgroups: {monitor.skipped_subgroups}")
    print(f"\nGlobal baseline violation: mean={monitor.global_baseline_mean:.4f}, "
          f"std={monitor.global_baseline_std:.4f}")
    print("Per-subgroup baseline mean violations:")
    for name in monitor.subgroup_names:
        print(f"   {name}: mean={monitor.baseline_means[name]:.4f}, "
              f"std={monitor.baseline_stds[name]:.4f}")

    # are baseline violations near-zero? (the sigma-bounds question):
    if monitor.global_baseline_mean < 1e-6:
        print("\n[!] WARNING: baseline violations are ~0. The 4-sigma bounds may "
              "be too loose to produce a usable signal. Consider percentile bounds.")
    else:
        print("\n[OK] Baseline violations are non-zero.")

    # train classifier (all features + protected, as in baseline): 
    # train WITH the protected attribute appended, identical to the validated baseline classifier. The runner appends each batch's 
    # protected column before predicting, so shapes match
    Xtr = np.column_stack([bundle.X_train, bundle.protected_train])
    model, threshold = train_model("lr", Xtr, bundle.y_train, seed=42)
    print(f"\nTrained LR (with protected col, as in baseline); "
          f"tuned threshold = {threshold:.3f}")

    # build one drift scenario:
    spec = DriftSpec(
        drift_type=DriftType.GROUP_COVARIATE,
        drift_pattern=DriftPattern.GRADUAL,
        onset_batch=5,
        magnitude=3.0,
        affected_group=0,  # minority
        feature_indices=None,  # all features (continuous portion is what the monitor sees)
        random_state=42,
    )
    stream = generate_stream(bundle, spec, n_batches=20, random_state=42)
    print(f"\nGenerated stream: {len(stream.batches)} batches, "
          f"batch_size={stream.batch_size}, onset at batch {spec.onset_batch}")

    # run the experiment:
    result = run_single_experiment(
        bundle=bundle, monitor=monitor, model=model, threshold=threshold,
        stream=stream, drift_label="gradual_group_min_mag3", model_name="lr",
    )

    # per-batch signal trace (do violations rise under drift?):
    print("\nPer-batch trace (drift onset at batch 5):")
    print(f"{'batch':>5} {'g_mean':>8} {'g_ewma':>8} {'sub_diff':>9} "
          f"{'max_dev':>8} {'DI*':>6} {'AOD*':>6} {'BalAcc':>7} {'alert':>6}")
    for r in result.batch_records:
        # find alert from the monitor trace is not stored per-record; recompute label:
        print(f"{r.batch_index:>5} {r.global_mean:>8.4f} {r.global_ewma:>8.4f} "
              f"{r.subgroup_violation_diff:>9.4f} {r.max_subgroup_deviation:>8.3f} "
              f"{r.DI_star:>6.3f} {r.AOD_star:>6.3f} {r.BalAcc:>7.3f}")

    # correlations (do they compute?):
    print("\nSpearman correlations (signal vs fairness metric):")
    print(f"{'signal':<38} {'DI*':>8} {'AOD*':>8} {'BalAcc':>8}")
    for sig in sorted(result.correlations.keys()):
        row = result.correlations[sig]
        di = row["DI_star"]["spearman"]
        aod = row["AOD_star"]["spearman"]
        ba = row["BalAcc"]["spearman"]
        print(f"{sig:<38} {di:>8.3f} {aod:>8.3f} {ba:>8.3f}")

    print("\n[done] If violations rise after batch 5 and correlations are "
          "non-NaN, the pipeline works end-to-end.")


if __name__ == "__main__":
    main()