"""
Violation scoring for Conformance Constraints.

Implements the quantitative-violation semantics from Fariha et al. (2021), Section 3.2, Equation 1:

    JΦK(t)        = Σ_k  q_k · Jφ_kK(t)                  (weighted sum over projections)
    Jφ_kK(t)      = η( dist_k(t) / σ_k )                 (per-projection violation)
    dist_k(t)     = max(0, F_k(t) − ub_k, lb_k − F_k(t)) (out-of-bounds distance)
    η(x)          = 1 − e^(−x)                           (normalization function)

where:
  - F_k(t) is the value of projection k applied to tuple t,
  - [lb_k, ub_k] are the learned bounds for projection k,
  - σ_k is the standard deviation of projection k on the fit data (this is the scaling factor α = 1/σ_k from Fariha Section 3.2,
    which standardizes distances so projections are comparable),
  - q_k is the normalized importance weight of projection k.

A violation of 0 means the tuple is fully inside all bounds (conforms). Higher values mean the tuple sits further outside the learned 
constraints. Each projection's contribution is bounded by its weight q_k (since η < 1), so the total violation is bounded by Σ q_k = 1.

This is a batch (vectorized) implementation; it scores every row of X against every projection at once using numpy broadcasting, 
no Python loops.
"""

import numpy as np

# compute total conformance-constraint violation for each row of X:
def compute_violation_batch(
    X: np.ndarray,  # (n_samples, n_features) data to score. Already standardized & containing only continuous columns
    projections: np.ndarray,  # (K, n_features) projection directions (rows)
    lower_bounds: np.ndarray,  # (K,) lower bound per projection
    upper_bounds: np.ndarray,  # (K,) upper bound per projection
    sigmas: np.ndarray,  # (K,) standard deviation per projection (for scaling)
    weights: np.ndarray,  # (K,) normalized importance weights (sum to 1)
) -> np.ndarray:

    # ensure X is a float numpy array regardless of what was passed in:
    X = np.asarray(X, dtype=float)

    # project every point onto every direction. 
    # Z[i, k] = F_k(point i) = projection k dotted with point i.
    # X is (n, d), projections.T is (d, K), so Z is (n, K):
    Z = X @ projections.T  # (n, K)

    # out-of-bounds distance, per Fariha: max(0, F − ub, lb − F). 
    # for each (point, projection): how far outside [lb, ub] is the projected value? 0 if inside. 
    # We compute both the "above upper" and "below lower" overflows and take the positive part.
    above = Z - upper_bounds  # (n, K), >0 if past upper
    below = lower_bounds - Z  # (n, K), >0 if past lower
    
    # element-wise max of (0, above, below): inside-bounds points give 0
    distance = np.maximum(0.0, np.maximum(above, below))  # (n, K)

    # per-projection violation: η(dist / σ) = 1 − exp(−dist / σ)
    # dividing by sigma is Fariha's scaling factor α = 1/σ (Section 3.2), making distances comparable across projections of diff spread:
    eta = 1.0 - np.exp(-distance / sigmas)  # (n, K)

    # weighted sum across projections: Σ_k q_k · η_k
    # multiply each projection's violation by its importance weight, then sum across projections (axis=1) to get one score per point:
    violations = (weights * eta).sum(axis=1)  # (n,)

    return violations # array of total violation scores