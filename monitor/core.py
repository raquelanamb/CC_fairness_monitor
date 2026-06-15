"""
FairnessDriftMonitor: the orchestrator of the CC-based fairness drift monitor.

This ties together the CC module (per-subgroup fingerprints) and stats.py (temporal tracking) into a working monitor. It has two phases:
1. SETUP (from_bundle): at baseline, slice to continuous features, standardize globally, split the training data into the four 
   (protected x label) subgroups, fit one ConformanceConstraints per subgroup (skipping any too small to model), and record baseline 
   violation statistics both per-subgroup and globally.

2. MONITORING (monitor_batch / monitor_stream): for each incoming batch (with NO protected attribute), standardize with the stored frame, 
   score against all subgroup CCs, assign each point to its minimum-violation subgroup, aggregate violation statistics, update the EWMA, 
   compute the Wasserstein divergence from the previous batch, and check control-limit alerts.

Standardization is performed ONCE here at the monitor level (a single frame over the full training set's continuous features) and applied 
identically to every subgroup and every serving batch, so violation scores are comparable across subgroups (which the min-violation 
assignment requires) and so that drift is preserved rather than normalized away batch-by-batch.

Baselines: a GLOBAL baseline (every training point scored at its min-violation subgroup) provides the aggregate violation signal that is 
correlated against fairness metrics. PER-SUBGROUP baselines support localization of drift to a particular (protected x label) subgroup. 
Both are recorded.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from cc.constraints import ConformanceConstraints
from .stats import compute_ewma, compute_divergence, check_alert


# Subgroup-size threshold: after density filtering keeps the densest 20 (Yang & Meliou), PCA needs ~5 points/dimension for stable 
# estimation, giving 0.2*n >= 5*d  =>  n >= 25*d. 
# POINTS_PER_DIM encodes the 25; ABSOLUTE_FLOOR is a conservative floor. These are deliberately conservative; the exact values are 
# not critical, only to exclude subgroups too sparse to model:
ABSOLUTE_FLOOR = 30
POINTS_PER_DIM = 25


# returns readable subgroup key:
def subgroup_name(protected: int, label: int) -> str:
    p = "min" if protected == 0 else "maj"  # protected: 0 = minority, 1 = majority
    y = "pos" if label == 1 else "neg"  # label: 0 = neg, 1 = pos
    return f"{p}_{y}"


# obj to hold violation stats for one subgroup within one batch:
@dataclass
class SubgroupStats:
    name: str
    mean: float
    std: float
    count: int


# obj to hold everything computed for a single monitored batch:
@dataclass
class BatchMonitorResult:
    batch_index: int  # which batch
    global_mean: float  # mean min-violation over all points
    global_std: float  # min-violation std over all points
    global_ewma: float  # EWMA of global_mean up to this batch
    divergence: Optional[float]  # Wasserstein vs previous batch (None for first)
    alert: bool  # global control-limit alert fired?
    subgroup_stats: Dict[str, SubgroupStats]  # per-assigned-subgroup stats this batch
    assignments: np.ndarray  # (n_points,) assigned subgroup index per point


# obj to hold full result of monitoring a stream of batches:
@dataclass
class MonitorResult:
    batches: List[BatchMonitorResult] = field(default_factory=list)
    subgroup_names: List[str] = field(default_factory=list)
    global_baseline_mean: float = 0.0
    global_baseline_std: float = 0.0
    baseline_means: Dict[str, float] = field(default_factory=dict)
    baseline_stds: Dict[str, float] = field(default_factory=dict)
    skipped_subgroups: List[str] = field(default_factory=list)


class FairnessDriftMonitor:
    def __init__(
        self,
        constraints: Dict[str, ConformanceConstraints],
        continuous_indices: List[int],
        mean: np.ndarray,
        std: np.ndarray,
        global_baseline_mean: float,
        global_baseline_std: float,
        baseline_means: Dict[str, float],
        baseline_stds: Dict[str, float],
        skipped_subgroups: List[str],
        ewma_alpha: float = 0.2,  # ECDD recommended value (Lu et al. 2019)
        alert_k: float = 3.0,  # 3-sigma control limit (SPC convention)
    ):
        self.constraints = constraints
        self.continuous_indices = continuous_indices
        self.mean = mean
        self.std = std
        self.global_baseline_mean = global_baseline_mean
        self.global_baseline_std = global_baseline_std
        self.baseline_means = baseline_means
        self.baseline_stds = baseline_stds
        self.skipped_subgroups = skipped_subgroups
        self.ewma_alpha = ewma_alpha
        self.alert_k = alert_k

        # fixed order of fitted subgroups (used to index the violation matrix):
        self.subgroup_names = list(constraints.keys())

        # temporal state (cleared by reset()):
        self._prev_violations: Optional[np.ndarray] = None
        self._ewma: Optional[float] = None
        self._batch_count = 0


    # helper funcs:
    def _standardize_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_cont = X[:, self.continuous_indices]  # slicing incoming full-feature batch to continuous cols
        return (X_cont - self.mean) / self.std  # standardizing with the stored global frame (NOT the batch's own statistics)

    def _score_all_subgroups(self, X_std: np.ndarray) -> np.ndarray:
        # score standardized points against every fitted subgroup CC:
        cols = [self.constraints[name].score(X_std) for name in self.subgroup_names]
        return np.column_stack(cols)   # returns (n_points, n_subgroups) violation matrix in subgroup_names order

    # setup:
    @classmethod
    def from_bundle(
        cls,
        bundle,
        cc_kwargs: Optional[dict] = None,
        ewma_alpha: float = 0.2,
        alert_k: float = 3.0,
    ) -> "FairnessDriftMonitor":
        cc_kwargs = cc_kwargs or {}

        # continuous-only training features:
        X = np.asarray(bundle.X_train, dtype=float)
        X_cont = X[:, bundle.continuous_indices]
        n_features = X_cont.shape[1]

        # global standardization frame (over ALL continuous training data):
        mean = X_cont.mean(axis=0)
        std = X_cont.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)  # guard zero-variance columns (avoid divide-by-zero):
        X_std = (X_cont - mean) / std

        protected = np.asarray(bundle.protected_train, dtype=int)
        y = np.asarray(bundle.y_train, dtype=int)

        min_subgroup_samples = max(ABSOLUTE_FLOOR, POINTS_PER_DIM * n_features)

        constraints: Dict[str, ConformanceConstraints] = {}
        baseline_means: Dict[str, float] = {}
        baseline_stds: Dict[str, float] = {}
        skipped: List[str] = []

        # fit a CC per subgroup (or skip), record per-subgroup stats:
        for p in (0, 1):
            for label in (0, 1):
                name = subgroup_name(p, label)
                mask = (protected == p) & (y == label)
                X_sub = X_std[mask]

                if X_sub.shape[0] < min_subgroup_samples:
                    skipped.append(name)
                    continue

                cc = ConformanceConstraints(**cc_kwargs)
                cc.fit(X_sub)
                constraints[name] = cc

                base_v = cc.score(X_sub)
                baseline_means[name] = float(base_v.mean())
                baseline_stds[name] = float(base_v.std())

        # GLOBAL baseline: score every training point at its min-violation subgroup (same mechanism used at monitoring time), so the global
        # baseline is computed consistently with how batches are scored:
        names = list(constraints.keys())
        if not names:
            raise RuntimeError("No subgroups could be fit (all below size threshold).")
        cols = [constraints[name].score(X_std) for name in names]
        viol_matrix = np.column_stack(cols)              # (n_train, n_subgroups)
        min_viol = viol_matrix.min(axis=1)               # per-point min violation
        global_baseline_mean = float(min_viol.mean())
        global_baseline_std = float(min_viol.std())

        return cls(
            constraints=constraints,
            continuous_indices=list(bundle.continuous_indices),
            mean=mean,
            std=std,
            global_baseline_mean=global_baseline_mean,
            global_baseline_std=global_baseline_std,
            baseline_means=baseline_means,
            baseline_stds=baseline_stds,
            skipped_subgroups=skipped,
            ewma_alpha=ewma_alpha,
            alert_k=alert_k,
        )

    # monitor a single incoming batch (full feature matrix, NO protected attr):
    def monitor_batch(self, X: np.ndarray) -> BatchMonitorResult:

        # standardize:
        X_std = self._standardize_batch(X)

        # score vs all subgroup CCs:
        viol_matrix = self._score_all_subgroups(X_std)  # (n_points, n_subgroups) violation matrix

        # min-violation assignment: each point's score is its smallest violation, and its assignment is the argmin subgroup. 
        # Note: this assigns by conformance, not by true (unobserved) group membership:
        min_viol = viol_matrix.min(axis=1)  # (n_points,)
        assignments = viol_matrix.argmin(axis=1)  # (n_points,) subgroup idx

        global_mean = float(min_viol.mean())
        global_std = float(min_viol.std())

        # aggregate per-assigned-subgroup stats for this batch:
        subgroup_stats: Dict[str, SubgroupStats] = {}
        for idx, name in enumerate(self.subgroup_names):
            sel = (assignments == idx)
            count = int(sel.sum())
            if count > 0:
                vals = min_viol[sel]
                subgroup_stats[name] = SubgroupStats(
                    name=name, mean=float(vals.mean()),
                    std=float(vals.std()), count=count,
                )
            else:
                # record empty subgroups too, so every batch has all subgroup keys:
                subgroup_stats[name] = SubgroupStats(name=name, mean=0.0, std=0.0, count=0)

        # update EWMA of the global mean violation (initialize to first batch's mean):
        if self._ewma is None:
            self._ewma = global_mean
        else:
            self._ewma = compute_ewma(global_mean, self._ewma, self.ewma_alpha)

        # compute Wasserstein divergence vs previous batch (None on the first batch):
        if self._prev_violations is None:
            divergence = None
        else:
            divergence = compute_divergence(min_viol, self._prev_violations)

        # divergence compares the distribution of per-point MIN violations between consecutive batches (not the full violation matrix 
        # or the means) — how the shape of "best-conformance" scores shifts batch-to-batch:
        self._prev_violations = min_viol

        # check control-limit alert- compare the EWMA against the GLOBAL baseline:
        alert = check_alert(
            current_mean=self._ewma,
            baseline_mean=self.global_baseline_mean,
            baseline_std=self.global_baseline_std,
            threshold=self.alert_k,
        )

        result = BatchMonitorResult(
            batch_index=self._batch_count,
            global_mean=global_mean,
            global_std=global_std,
            global_ewma=self._ewma,
            divergence=divergence,
            alert=alert,
            subgroup_stats=subgroup_stats,
            assignments=assignments,
        )
        self._batch_count += 1
        return result

    # monitor an ordered list of batches and return the aggregated result:
    def monitor_stream(self, batches: List[np.ndarray]) -> MonitorResult:
        self.reset()
        result = MonitorResult(
            subgroup_names=list(self.subgroup_names),
            global_baseline_mean=self.global_baseline_mean,
            global_baseline_std=self.global_baseline_std,
            baseline_means=dict(self.baseline_means),
            baseline_stds=dict(self.baseline_stds),
            skipped_subgroups=list(self.skipped_subgroups),
        )
        for X in batches:
            result.batches.append(self.monitor_batch(X))
        return result

    # clear temporal state so monitor can be reused on a fresh stream:
    def reset(self) -> None:
        self._prev_violations = None
        self._ewma = None
        self._batch_count = 0