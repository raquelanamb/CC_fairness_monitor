"""
Temporal stream generator for drift simulation (thesis Objective 3).

Core idea: the test set is the sampling pool (it was held out during CC training), so sampling batches from it simulates temporal windows
(T0, T1, T2, ...) drawn from the deployment distribution. Batch 0 is always clean (the deployment baseline). Starting at onset_batch, drift 
is injected with per-batch magnitude controlled by the DriftPattern.

Feature statistics (means, stds) used to scale drift are computed from the TRAINING set, so drift magnitude is expressed relative to the 
distribution the monitor learned its baseline on, which is consistent with how the monitor standardizes.
"""

import numpy as np
from typing import Optional, List

from .config import DriftSpec, DriftType, DriftPattern, BatchResult, StreamResult
from .injectors import apply_mean_shift, apply_variance_scaling


# generate a temporal stream of (possibly drifted) batches from a bundle:
def generate_stream(
    bundle,  # a DatasetBundle (uses X_test / y_test / protected_test as the sampling pool; X_train for drift-scaling statistics)
    drift_spec: DriftSpec,  # what drift to inject and when
    n_batches: int = 20,  # number of temporal batches (default 20, matching the seed count and the batch-size arithmetic)
    batch_size: Optional[int] = None,  # samples per batch; default len(X_test) // n_batches
    random_state: int = 42,  # seed for the batch sampling
) -> StreamResult:

    if drift_spec.onset_batch >= n_batches:
        raise ValueError(
            f"onset_batch ({drift_spec.onset_batch}) must be < n_batches ({n_batches})"
        )
    if (drift_spec.drift_type == DriftType.GROUP_COVARIATE
            and drift_spec.affected_group is None):
        raise ValueError("affected_group must be specified for GROUP_COVARIATE drift")

    X_train = np.asarray(bundle.X_train, dtype=float)
    feature_stds = np.maximum(np.std(X_train, axis=0), 1e-8)   # guard zero-std
    feature_means = np.mean(X_train, axis=0)

    X_test = np.asarray(bundle.X_test, dtype=float)
    y_test = np.asarray(bundle.y_test)
    prot_test = np.asarray(bundle.protected_test, dtype=int)

    rng = np.random.RandomState(random_state)
    n_test = len(X_test)
    if batch_size is None:
        batch_size = n_test // n_batches

    batches: List[BatchResult] = []

    for batch_idx in range(n_batches):
        # sample a batch from the test pool (without replacement if possible):
        replace = n_test < batch_size
        indices = rng.choice(n_test, size=batch_size, replace=replace)

        X_batch = X_test[indices].copy()
        y_batch = y_test[indices].copy()
        prot_batch = prot_test[indices].copy()

        # how strong is the drift in THIS batch (per the temporal pattern):
        eff = _compute_effective_magnitude(batch_idx, drift_spec, n_batches)

        # apply drift unless it is a no-op (0 shift, or scale factor of 1.0):
        is_noop = (
            (drift_spec.drift_pattern == DriftPattern.VARIANCE and eff <= 1.0)
            or (drift_spec.drift_pattern != DriftPattern.VARIANCE and eff <= 0.0)
        )
        if not is_noop:
            X_batch = _apply_drift_to_batch(
                X=X_batch, protected=prot_batch, drift_spec=drift_spec,
                effective_magnitude=eff, feature_stds=feature_stds,
                feature_means=feature_means,
            )
            drift_applied = True
        else:
            drift_applied = False

        batches.append(BatchResult(
            batch_index=batch_idx,
            X=X_batch, y=y_batch, protected=prot_batch,
            drift_applied=drift_applied,
            drift_type=drift_spec.drift_type if drift_applied else None,
            drift_pattern=drift_spec.drift_pattern if drift_applied else None,
            drift_magnitude_actual=eff,
            affected_group=drift_spec.affected_group if drift_applied else None,
        ))

    return StreamResult(
        batches=batches,
        drift_spec=drift_spec,
        dataset_name=bundle.dataset_name,
        n_batches=n_batches,
        batch_size=batch_size,
        feature_stds=feature_stds,
        feature_means=feature_means,
    )


# per-batch drift magnitude given the temporal pattern:
#  - before onset: 0.0 (or 1.0 for VARIANCE, i.e. no scaling)
#  - ABRUPT:   full magnitude immediately at/after onset
#  - GRADUAL:  linear ramp 0 -> magnitude across the post-onset batches
#  - VARIANCE: linear ramp 1.0 -> magnitude across the post-onset batches
def _compute_effective_magnitude(batch_idx, drift_spec, n_batches) -> float:

    if batch_idx < drift_spec.onset_batch:
        return 1.0 if drift_spec.drift_pattern == DriftPattern.VARIANCE else 0.0

    if drift_spec.drift_pattern == DriftPattern.ABRUPT:
        return drift_spec.magnitude

    batches_since_onset = batch_idx - drift_spec.onset_batch
    batches_of_drift = n_batches - drift_spec.onset_batch - 1
    progress = 1.0 if batches_of_drift <= 0 else batches_since_onset / batches_of_drift

    if drift_spec.drift_pattern == DriftPattern.GRADUAL:
        return drift_spec.magnitude * progress
    elif drift_spec.drift_pattern == DriftPattern.VARIANCE:
        return 1.0 + (drift_spec.magnitude - 1.0) * progress
    else:
        raise ValueError(f"Unknown drift pattern: {drift_spec.drift_pattern}")


# apply drift to a batch: GLOBAL_COVARIATE: modify all rows. GROUP_COVARIATE: modify only rows where protected == affected_group
def _apply_drift_to_batch(
    X, protected, drift_spec, effective_magnitude, feature_stds, feature_means
) -> np.ndarray:

    feat_idx = drift_spec.feature_indices
    if feat_idx is not None:
        feat_idx = np.array(feat_idx)

    def _drift(arr):
        if drift_spec.drift_pattern == DriftPattern.VARIANCE:
            return apply_variance_scaling(arr, effective_magnitude, feature_means, feat_idx)
        return apply_mean_shift(arr, effective_magnitude, feature_stds, feat_idx)

    if drift_spec.drift_type == DriftType.GLOBAL_COVARIATE:
        return _drift(X)

    elif drift_spec.drift_type == DriftType.GROUP_COVARIATE:
        mask = (protected == drift_spec.affected_group)
        if mask.sum() == 0:
            return X
        X[mask] = _drift(X[mask])
        return X

    raise ValueError(f"Unknown drift type: {drift_spec.drift_type}")