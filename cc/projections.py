"""
Projection generation for Conformance Constraints.

Faithfully implements the projection-discovery procedure from Fariha et al. (2021), 
"Conformance Constraint Discovery: Measuring Trust in Data-Driven Systems", Algorithm 1 (lines 1-6).

The key theoretical result (Fariha et al. Theorem 12, 13): the eigenvectors of D'_N^T @ D'_N give projection directions that 
are mutually uncorrelated (orthogonal) and span the full range from lowest to highest variance. 
The LOW-variance projections construct the strongest conformance constraints (Section 4.1.2), so we keep all K projections 
and let constraints.py weight the low-variance ones more heavily (Algorithm 1 line 7).

- Theorem 12: proves that highly-correlated projections can be combined into a single lower-variance projection that makes 
  a stronger constraint. In other words, redundant directions can be merged into better ones.
- Theorem 13: proves that Algorithm 1 (the PCA-based procedure) actually produces the optimal set of projections (they're 
  uncorrelated and include the lowest-variance one).

This module implements ONLY the discovery of projection directions (Algorithm 1 lines 1-6). The importance 
weights (line 7), bounds (Section 4.1.1), and normalization (line 8) are computed in constraints.py, because they 
all depend on the projection standard deviations which are naturally computed there alongside the bounds and violation scoring.
"""

import numpy as np

# generate projection directions following Fariha et al. (2021) Algorithm 1, lines 2-6
# (non-numerical attributes must already be dropped, which is Algorithm 1 line 1, handled before calling this function):
def generate_pca_projections(X: np.ndarray) -> np.ndarray:

    # makes sure X is numpy array of floats:
    X = np.asarray(X, dtype=float)

    # unpack dimensions into variables:
    n_samples, n_features = X.shape

    # --- Algorithm 1, line 2: D'_N <- [1; D_N] ---
    # Prepend a column of 1s. Fariha et al. add this constant column so the eigenvectors can capture an additive constant 
    # within each projection, which (per the paper) makes the approach work even for unnormalized data. The coefficient 
    # learned for this constant column is removed in line 5 below:
    ones = np.ones((n_samples, 1))
    D_prime = np.hstack([ones, X])          # shape (n_samples, n_features + 1)

    # --- Algorithm 1, line 3: {w_1, ..., w_K} <- eigenvectors of D'_N^T · D'_N ---
    # D_prime.T @ D_prime is a square, symmetric (n_features+1, n_features+1) matrix (the "scatter matrix" - each entry 
    # captures a relationship between two columns (features)). Its eigenvectors are the projection directions. I use 
    # np.linalg.eigh (not eig) because the matrix is symmetric: eigh is more accurate and returns REAL eigenvalues/vectors 
    # sorted in ASCENDING order of eigenvalue (the eigenvalue corresponds to the variance along that eigenvector's direction; 
    # so ascending order means the lowest-variance projections come first, and the highest-variance ones come last):
    gram = D_prime.T @ D_prime       # shape (n_features+1, n_features+1)          ( @ is matrix multiplication )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    # eigh returns eigenvectors as COLUMNS of the returned matrix, so eigenvectors[:, k] is the k-th eigenvector w_k. 
    # We transpose so each ROW is one eigenvector, which is easier to iterate over:
    W_full = eigenvectors.T      # shape (n_features+1, n_features+1)

    # --- Algorithm 1, lines 4-6: build normalized projections ---
    projections = []
    for w_k in W_full: # (line 4)

        # line 5: w'_k <- w_k with first element removed (drop the coefficient for the constant column added in line 2):
        w_prime = w_k[1:]      # shape (n_features,)

        # line 6: F_k <- λA: (A^T · w'_k) / ||w'_k||   (normalize)
        # i.e. normalize the projection direction to unit length. The division by ||w'_k|| is what makes projections 
        # comparable across directions (a projection's raw scale is otherwise arbitrary):
        norm = np.linalg.norm(w_prime)
        
        # degenerate case - the eigenvector was essentially all in the constant-column coefficient; skip it (no usable direction):
        if norm < 1e-12:
            continue

        # add normalized projection direction to list:
        projections.append(w_prime / norm)

    # stack into a (K, n_features) matrix, one projection direction per row:
    W = np.vstack(projections)   # typically, shape = (K, n_features) = (n_features, n_features)

    return W



