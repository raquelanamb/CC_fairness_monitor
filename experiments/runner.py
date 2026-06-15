"""
Single correlation experiment: (dataset x drift_spec x model).

Pipeline for one experiment cell:
  1. fit the FairnessDriftMonitor on training data (done by caller, passed in)
  2. train the classifier on training data (done by caller, passed in)
  3. generate a drifted stream from the test pool
  4. for each batch: run the monitor (CC signals) AND predict with the classifier to compute per-batch fairness metrics (DI*, AOD*, BalAcc)
  5. correlate each CC signal (and the KS / KL baseline detectors) against each fairness metric, across batches, using Spearman (primary) 
     and Pearson.

CC SIGNALS (8):
  global:   global_mean, global_ewma, global_std, subgroup_violation_diff
  subgroup: per_subgroup_ewma (per subgroup), max_subgroup_deviation, per_subgroup_baseline_relative (per subgroup),
            assignment_proportion_drift
BASELINE DETECTORS (generic, for comparison; Objective 7): ks_statistic, kl_divergence (train vs batch, over continuous features)

Correlation: Spearman is primary (the hypothesis is a MONOTONIC relationship between violation and fairness degradation, not a linear one; 
Spearman makes no linearity assumption and is robust to outliers such as Credit's unstable DI). 
Pearson is reported secondarily for completeness.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

from experiments.fairness import compute_metrics
from experiments.models import predict


# the following dataclasses are result containers:

# per-batch fairness metrics + CC signals + baseline detectors:
@dataclass
class BatchRecord:
    batch_index: int
    # fairness (transformed: higher = fairer):
    DI_star: float
    AOD_star: float
    BalAcc: float
    # global CC signals:
    global_mean: float
    global_ewma: float
    global_std: float
    subgroup_violation_diff: float
    # subgroup CC signals:
    per_subgroup_ewma: Dict[str, float]
    max_subgroup_deviation: float
    per_subgroup_baseline_relative: Dict[str, float]
    assignment_proportion_drift: float
    # baseline detectors:
    ks_statistic: float
    kl_divergence: float


@dataclass
class ExperimentResult:
    dataset_name: str
    drift_label: str
    model_name: str
    n_batches: int
    batch_records: List[BatchRecord] = field(default_factory=list)
    # correlations[signal][metric] = {"spearman": r, "spearman_p": p, "pearson": r, "pearson_p": p}
    correlations: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)


# the following funcs define generic baseline drift detectors (for comparison; Objective 7):

# These are the generic, subgroup-blind drift detectors the CC monitor is compared against. Two deliberate choices make the comparison fair:
# 1) train-vs-batch reference: each batch is compared against the TRAINING set, the same fixed baseline the CC monitor references. Comparing
#    batch-to-batch instead would measure a different quantity; anchoring both detector families to the training distribution keeps them
#    measuring drift from the same reference.
# 2) continuous features only: the baselines see the SAME feature subset as the CC monitor (bundle.continuous_indices). Giving the baselines 
#    all features while the monitor sees only continuous ones would be an unfair comparison on unequal information; restricting both to 
#    continuous features isolates the question of WHICH detector better tracks fairness drift, not which sees more features.

# mean per-feature 2-sample KS statistic between reference and batch:
def _ks_statistic(X_ref: np.ndarray, X_batch: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    d = X_ref.shape[1]
    stats = [ks_2samp(X_ref[:, j], X_batch[:, j]).statistic for j in range(d)]
    return float(np.mean(stats))

# mean per-feature histogram KL divergence KL(batch || ref):
def _kl_divergence(X_ref: np.ndarray, X_batch: np.ndarray, bins: int = 20) -> float:
    d = X_ref.shape[1]
    kls = []
    for j in range(d):
        lo = min(X_ref[:, j].min(), X_batch[:, j].min())
        hi = max(X_ref[:, j].max(), X_batch[:, j].max())
        if hi <= lo:
            kls.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(X_batch[:, j], bins=edges, density=True)
        q, _ = np.histogram(X_ref[:, j], bins=edges, density=True)
        # smooth to avoid log(0) / divide-by-zero:
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        kls.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(kls))


# the runner:

# run one correlation experiment. The monitor and model are pre-fit; the stream is pre-generated (so the caller controls seeds/scenarios):
def run_single_experiment(
    bundle,  # DatasetBundle (for continuous_indices, protected, train stats)
    monitor,  # a fitted FairnessDriftMonitor
    model,  # a trained classifier
    threshold: float,  # decision threshold for the classifier
    stream,  # a StreamResult from generate_stream
    drift_label: str,
    model_name: str,
) -> ExperimentResult:

    monitor.reset()
    cont = bundle.continuous_indices

    # reference (training) continuous features for the KS / KL baselines:
    X_train_cont = np.asarray(bundle.X_train, dtype=float)[:, cont]

    # baseline assignment proportions (training data), for assignment-drift:
    base_props = _baseline_assignment_proportions(monitor, bundle)

    # per-subgroup running EWMA state (initialized lazily on first batch):
    sub_ewma: Dict[str, Optional[float]] = {n: None for n in monitor.subgroup_names}
    ewma_alpha = monitor.ewma_alpha

    records: List[BatchRecord] = []

    for batch in stream.batches:
        # monitor - CC signals for this batch:
        mon = monitor.monitor_batch(batch.X)

        # global signals come straight off the BatchMonitorResult:
        global_mean = mon.global_mean
        global_ewma = mon.global_ewma
        global_std = mon.global_std

        # subgroup_violation_diff = minority-assigned mean - majority-assigned mean:
        sgd = _subgroup_violation_diff(mon)

        # per-subgroup EWMA (maintained here; monitor only EWMAs the global):
        per_sub_ewma = {}
        for name in monitor.subgroup_names:
            cur = mon.subgroup_stats[name].mean
            if sub_ewma[name] is None:
                sub_ewma[name] = cur
            else:
                sub_ewma[name] = ewma_alpha * cur + (1 - ewma_alpha) * sub_ewma[name]
            per_sub_ewma[name] = sub_ewma[name]

        # per-subgroup deviation from own baseline, normalized by baseline std:
        # normalizing by each subgroup's OWN baseline std makes deviations comparable across subgroups: a raw shift of 0.5 means more for a
        # tightly-distributed subgroup than a loosely-distributed one, so the max-over-subgroups is only meaningful once each is put on a 
        # common (z-score-like) scale
        per_sub_baseline_rel = {}
        deviations = []
        for name in monitor.subgroup_names:
            cur = mon.subgroup_stats[name].mean
            base_m = monitor.baseline_means[name]
            base_s = monitor.baseline_stds[name]
            per_sub_baseline_rel[name] = cur - base_m
            if base_s > 1e-12:
                deviations.append((cur - base_m) / base_s)
        max_sub_dev = float(max(deviations)) if deviations else 0.0

        # assignment-proportion drift: total-variation distance between this batch's assignment proportions and the baseline (training) 
        # proportions. TV distance = 0.5 * sum |p_i - q_i| is a clean, bounded-[0,1] summary of how much the WHERE-points-conform 
        # distribution has shifted, which is a distinct drift facet from how MUCH points violate:
        cur_props = _current_assignment_proportions(mon, monitor.subgroup_names)
        assign_drift = 0.5 * sum(
            abs(cur_props[n] - base_props[n]) for n in monitor.subgroup_names
        )

        # baseline detectors (continuous features only):
        X_batch_cont = np.asarray(batch.X, dtype=float)[:, cont]
        ks = _ks_statistic(X_train_cont, X_batch_cont)
        kl = _kl_divergence(X_train_cont, X_batch_cont)

        # classifier - fairness metrics for this batch:
        # The classifier was trained on ALL features WITH the protected attribute appended (matching ConFair's sensi_col_in_training=True 
        # and the baseline-validation setup). The drift stream carries only bundle features, so we append the batch's protected column 
        # before predicting, keeping this classifier identical to the validated baseline one. The protected attribute is used here as a 
        # classifier feature; it is used SEPARATELY (below) as the grouping variable for the fairness metrics.
        # (This does not involve the CC monitor, which never sees protected)
        X_batch_clf = np.column_stack([batch.X, batch.protected])
        y_pred = predict(model, X_batch_clf, threshold=threshold)
        fair = compute_metrics(batch.y, y_pred, batch.protected)

        records.append(BatchRecord(
            batch_index=batch.batch_index,
            DI_star=fair["DI_star"], AOD_star=fair["AOD_star"], BalAcc=fair["BalAcc"],
            global_mean=global_mean, global_ewma=global_ewma, global_std=global_std,
            subgroup_violation_diff=sgd,
            per_subgroup_ewma=per_sub_ewma,
            max_subgroup_deviation=max_sub_dev,
            per_subgroup_baseline_relative=per_sub_baseline_rel,
            assignment_proportion_drift=assign_drift,
            ks_statistic=ks, kl_divergence=kl,
        ))

    correlations = _compute_correlations(records, monitor.subgroup_names)

    return ExperimentResult(
        dataset_name=bundle.dataset_name,
        drift_label=drift_label,
        model_name=model_name,
        n_batches=len(stream.batches),
        batch_records=records,
        correlations=correlations,
    )


# ---- helpers -----------------------------------------------------------------

def _minority_majority_names(subgroup_names):
    """Split subgroup names into minority (min_*) and majority (maj_*)."""
    minority = [n for n in subgroup_names if n.startswith("min_")]
    majority = [n for n in subgroup_names if n.startswith("maj_")]
    return minority, majority


# minority-assigned mean violation - majority-assigned mean violation:
# Count-weighted across the two minority and two majority subgroups. Uses the monitor's conformance assignments only (label-free).
# Count-weighting (rather than a plain average of subgroup means) ensures a sparsely-populated subgroup in a given batch does not swing the 
# difference as much as a heavily-populated one; the weighted mean reflects the actual distribution of points across subgroups in that batch
def _subgroup_violation_diff(mon) -> float:

    minority, majority = _minority_majority_names(mon.subgroup_stats.keys())

    def weighted_mean(names):
        num = sum(mon.subgroup_stats[n].mean * mon.subgroup_stats[n].count for n in names)
        den = sum(mon.subgroup_stats[n].count for n in names)
        return num / den if den > 0 else 0.0

    return weighted_mean(minority) - weighted_mean(majority)


def _current_assignment_proportions(mon, subgroup_names) -> Dict[str, float]:
    counts = {n: mon.subgroup_stats[n].count for n in subgroup_names}
    total = sum(counts.values())
    return {n: (counts[n] / total if total > 0 else 0.0) for n in subgroup_names}


# assignment proportions on the TRAINING data (the baseline reference):
# the training data is used as the reference (rather than batch 0 of the stream) so the assignment-drift signal is anchored to the same 
# baseline frame the monitor uses for everything else (standardization frame, baseline means/stds). This keeps all drift measured relative 
# to one consistent reference distribution
def _baseline_assignment_proportions(monitor, bundle) -> Dict[str, float]:
    X = np.asarray(bundle.X_train, dtype=float)[:, bundle.continuous_indices]
    X_std = (X - monitor.mean) / monitor.std
    cols = [monitor.constraints[n].score(X_std) for n in monitor.subgroup_names]
    viol = np.column_stack(cols)
    assign = viol.argmin(axis=1)
    total = len(assign)
    props = {}
    for idx, name in enumerate(monitor.subgroup_names):
        props[name] = float((assign == idx).sum()) / total if total > 0 else 0.0
    return props


# correlate each CC signal and baseline detector against each fairness metric, across batches. Spearman (primary) and Pearson (secondary):
def _compute_correlations(records, subgroup_names):
    metrics = ["DI_star", "AOD_star", "BalAcc"]

    # assemble per-signal series across batches:
    series: Dict[str, List[float]] = {
        "global_mean": [r.global_mean for r in records],
        "global_ewma": [r.global_ewma for r in records],
        "global_std": [r.global_std for r in records],
        "subgroup_violation_diff": [r.subgroup_violation_diff for r in records],
        "max_subgroup_deviation": [r.max_subgroup_deviation for r in records],
        "assignment_proportion_drift": [r.assignment_proportion_drift for r in records],
        "ks_statistic": [r.ks_statistic for r in records],
        "kl_divergence": [r.kl_divergence for r in records],
    }
    # per-subgroup signals expand into one series per subgroup:
    for name in subgroup_names:
        series[f"per_subgroup_ewma[{name}]"] = [r.per_subgroup_ewma[name] for r in records]
        series[f"per_subgroup_baseline_relative[{name}]"] = [
            r.per_subgroup_baseline_relative[name] for r in records
        ]

    metric_series = {m: [getattr(r, m) for r in records] for m in metrics}

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sig_name, sig_vals in series.items():
        out[sig_name] = {}
        for m in metrics:
            out[sig_name][m] = _safe_corr(sig_vals, metric_series[m])
    return out


# Spearman + Pearson with guards for constant/degenerate series:
# Correlation is undefined when either series has no variation, so a constant series returns NaN. This is EXPECTED, not missing data 
# (e.g. the no-drift control produces flat signals, and a fairness metric can be constant across batches). NaN entries in the results table 
# are the correct outcome in those cases and should be read as 'correlation undefined', not 'computation failed':
def _safe_corr(x, y) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = {"spearman": np.nan, "spearman_p": np.nan,
              "pearson": np.nan, "pearson_p": np.nan}
    # need variation in both series for correlation to be defined:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return result
    try:
        sr, sp = spearmanr(x, y)
        result["spearman"], result["spearman_p"] = float(sr), float(sp)
    except Exception:
        pass
    try:
        pr, pp = pearsonr(x, y)
        result["pearson"], result["pearson_p"] = float(pr), float(pp)
    except Exception:
        pass
    return result