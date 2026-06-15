"""
Drift injection primitives for temporal simulation.

Each function takes a feature matrix and returns a MODIFIED COPY with drift applied. These are pure mathematical operations; the stream 
generator (stream.py) decides when and how strongly to call them.

Drift magnitudes are expressed in units of the training-set feature standard deviation, so a magnitude of 1.0 shifts a feature by one 
training std dev. This keeps drift strength comparable across features on different scales.
"""

import numpy as np
from typing import Optional


# shift the mean of selected features by 'magnitude' training std devs. used by the GRADUAL and ABRUPT patterns:
def apply_mean_shift(
    X: np.ndarray,  # (n, d) feature matrix; not modified in place
    magnitude: float,  # shift in units of feature std dev
    feature_stds: np.ndarray,  # (d,) per-feature training std devs
    feature_indices: Optional[np.ndarray] = None,  # which columns to shift; None = all
) -> np.ndarray:

    X_shifted = X.copy()
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])
    for j in feature_indices:
        X_shifted[:, j] += magnitude * feature_stds[j]
    return X_shifted


# Scale the spread of selected features around their training mean, leaving the mean unchanged:  
# X'[j] = mean[j] + scale_factor * (X[j] - mean[j]). Used by the VARIANCE pattern:
def apply_variance_scaling(
    X: np.ndarray,
    scale_factor: float,  # 1.0 = no change, 2.0 = doubled std dev
    feature_means: np.ndarray,  # (d,) per-feature training means
    feature_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    X_scaled = X.copy()
    if feature_indices is None:
        feature_indices = np.arange(X.shape[1])
    for j in feature_indices:
        deviation = X_scaled[:, j] - feature_means[j]
        X_scaled[:, j] = feature_means[j] + scale_factor * deviation
    return X_scaled