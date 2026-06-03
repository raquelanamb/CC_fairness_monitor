"""
Conformance Constraints learning and violation scoring.

Implements the CC discovery and quantitative-violation semantics from:
  - Fariha et al. (2021), "Conformance Constraint Discovery", Algorithm 1 (projections + importance weights) & Section 4.1.1 
    (sigma bounds)
  - Yang & Meliou (2024), ConFair, Algorithm 3 (density-based optimization) & the density parameters from their LearnCCrules.py

A ConformanceConstraints object learns a distributional "fingerprint" of a subgroup from training data, then scores how much 
new data violates that fingerprint. Higher violation = further from the subgroup's learned normal.

INPUT CONVENTIONS:
  1. This class receives a feature matrix containing ONLY continuous columns (selected upstream via bundle.continuous_indices, 
     per ConFair's num_threshold=8 rule). All columns are treated as continuous.
  2. This class receives ALREADY-STANDARDIZED data. Standardization is done ONCE at the monitor level using the full training 
     set's mean/std, then the same frame is applied to every subgroup and to serving batches. This follows ConFair's 
     LearnCCrules.py, which standardizes the whole training set's continuous columns up front (cc_df[cc_cols] = (cc_df - mean) / std)
     BEFORE splitting into subgroups. A shared standardization frame keeps violation scores comparable ACROSS subgroups, which 
     matters because the monitor assigns each serving point to its minimum-violation subgroup.
"""

import numpy as np
from .projections import generate_pca_projections
from .violation import compute_violation_batch
from sklearn.neighbors import KernelDensity


class ConformanceConstraints:

    def __init__(
        self,

        # C in the sigma bounds [mu - C*sigma, mu + C*sigma]. Fariha et al. Section 4.1.1 set C = 4 (99.99% of a normal 
        # distribution lies within 4 sigma of the mean):
        bound_factor: float = 4.0,

        # whether to apply the density-based optimization (Yang & Meliou Algorithm 3). Toggleable so the CONFAIR vs 
        # CONFAIR_0 ablation (their Figure 13) can be reproduced:
        density_filter: bool = True,

        # fraction of densest points to KEEP per subgroup (Yang & Meliou k = 0.2 * n; their dense_n = 0.2):
        density_keep_frac: float = 0.2,

        # KDE bandwidth (Yang & Meliou's dense_h = 0.1):
        density_bandwidth: float = 0.1,

        # KDE kernel (their dense_kernal = 'gaussian'):
        density_kernel: str = "gaussian",
    ):
    
        self.bound_factor = bound_factor
        self.density_filter = density_filter
        self.density_keep_frac = density_keep_frac
        self.density_bandwidth = density_bandwidth
        self.density_kernel = density_kernel

        # learned during fit():
        self.projections_ = None  # (K, n_features) projection directions
        self.sigmas_ = None  # (K,) std of each projection on fit data
        self.lower_bounds_ = None  # (K,) lower bound per projection
        self.upper_bounds_ = None  # (K,) upper bound per projection
        self.weights_ = None  # (K,) normalized importance weights
        self.is_fitted_ = False


    # Yang & Meliou Algorithm 3: keep only the densest points before learning constraints, so the bounds reflect the subgroup's core 
    # distribution and are not loosened by sparse outliers. Uses kernel density estimation, then keeps the top int(density_keep_frac * n) 
    # points by density (their k = 0.2 * n).
    # Note: This affects only WHICH points the projections/bounds are learned from. Scoring is still applied to all serving points. 
    # This matches ConFair's LearnCCrules.py: constraints are learned on cc_input (the dense head) but evaluated on the full data:
    def _density_filter(self, X: np.ndarray) -> np.ndarray:

        # X.shape is (rows, columns), so [0] grabs number of rows:
        n = X.shape[0]

        # computing how many points to keep (following Yang and Meliou's 20%):
        keep = int(self.density_keep_frac * n) 

        # if too few points to filter meaningfully, skip:
        if keep < 1 or keep >= n:
            return X

        # creates a kernel density estimator w/ parameters based on Yang and Meliou (bandwidth 0.1, gaussian kernel). 
        # KDE estimates how "dense" the data is at any given point (how many neighbors are nearby):
        kde = KernelDensity(bandwidth=self.density_bandwidth, kernel=self.density_kernel)

        # fit the KDE to the data, learning where the dense & sparse regions are:
        kde.fit(X)

        # for each point, compute log of its estimated density. Higher value = that point sits in a denser region (more neighbors 
        # around it). It returns log-density rather than raw density for numerical stability, but the ordering is the same (higher 
        # still means denser):
        log_density = kde.score_samples(X)

        # indices of the densest 'keep' points ordered by descending density:
        densest_idx = np.argsort(log_density)[::-1][:keep]

        # return the corresponding rows from X (only the densest points):
        return X[densest_idx]


    # learn conformance constraints from a subgroup's continuous features:
    def fit(self, X: np.ndarray) -> "ConformanceConstraints":

        # ensure X is a float numpy array regardless of what was passed in:
        X = np.asarray(X, dtype=float)

        # density filter (Yang & Meliou Algorithm 3), if enabled:
        if self.density_filter:
            X_fit = self._density_filter(X)
        else:
            X_fit = X

        # discover projection directions (Fariha Alg. 1 lines 2-6):
        self.projections_ = generate_pca_projections(X_fit)   # (K, n_features)

        # project the (filtered) data onto each direction:
        Z = X_fit @ self.projections_.T   # Z[i, k] = projection k applied to point i, shape: (n_fit, K)

        # per-projection mean & sigma:
        means = Z.mean(axis=0)  # (K,)
        self.sigmas_ = Z.std(axis=0)  # (K,)

        # guard zero-variance projections:
        self.sigmas_ = np.where(self.sigmas_ < 1e-12, 1e-12, self.sigmas_)

        # sigma bounds (Fariha Section 4.1.1, C = bound_factor):
        self.lower_bounds_ = means - self.bound_factor * self.sigmas_
        self.upper_bounds_ = means + self.bound_factor * self.sigmas_

        # importance weights (Fariha Alg. 1 line 7):
        # gamma_k = 1 / log(2 + sigma_k): lower-variance projections get HIGHER weight because they construct stronger 
        # (more discerning) constraints:
        raw_weights = 1.0 / np.log(2.0 + self.sigmas_)

        # normalize to sum to 1 (Alg. 1 line 8: divide by Z = sum of gammas):
        self.weights_ = raw_weights / raw_weights.sum()

        # flag that fitting is done, so score knows it's safe to run:
        self.is_fitted_ = True

        # return the fitted CC model (ConformanceConstraints instance, now carrying all learned projections, bounds, sigmas, & weights)
        return self


    # compute total violation score for each row of X:
    def score(self, X: np.ndarray) -> np.ndarray:

        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before score().")

        # ensure X is a float numpy array regardless of what was passed in:
        X = np.asarray(X, dtype=float)

        # delegate the per-projection violation math to violation.py:
        return compute_violation_batch(
            X=X,
            projections=self.projections_,
            lower_bounds=self.lower_bounds_,
            upper_bounds=self.upper_bounds_,
            sigmas=self.sigmas_,
            weights=self.weights_,
        )