"""
Statistical utilities for temporal fairness-drift monitoring.

These are the temporal-tracking tools the monitor applies to the CC violation signal. They are standard, literature-grounded 
drift-detection mechanisms; the novelty of this thesis is the SIGNAL being tracked (CC violations), not the trackers themselves. 
Each tool is justified below.

  - compute_ewma:   Exponentially Weighted Moving Average. Tracks the LEVEL of violation over time, smoothing batch-to-batch noise. 
                    The EWMA control chart is an established drift detector: the ECDD method (EWMA for Concept Drift Detection) applies 
                    exactly this mechanism to a classifier's error rate (reviewed in Lu et al. 2019; see also SPC-based drift detection 
                    in Gama et al. 2014). I apply it instead to the label-free CC violation signal. ECDD's authors recommend a weight 
                    of 0.2, which I adopt as a principled default.

  - compute_divergence: Wasserstein (earth-mover's) distance between consecutive batches' violation distributions. Tracks change in the 
                        SHAPE of the violation distribution, which the EWMA mean can miss (e.g. the mean is stable but the distribution
                        splits or spreads). Wasserstein is an established measure for unsupervised, distribution-based drift / change-point 
                        detection (Faber et al. 2021, WATCH). I use the MEASURE, not any specific algorithm.
                        Chosen over KL/JS because it operates directly on samples (no density estimation) and stays finite under 
                        non-overlapping supports.

  - check_alert:    Control-limit alert: fire when the (EWMA-smoothed) mean exceeds baseline_mean + threshold * baseline_std. 
                    This is the control-chart rule used by ECDD (warning/drift limits at multiples of sigma above baseline; Lu et al. 2019) 
                    and classical SPC (Gama et al. 2014).
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Optional

# calculates Exponentially Weighted Moving Average: EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
def compute_ewma(
    current_value: float,  # current batch observation x_t
    previous_ewma: float,  # previous EWMA value EWMA_{t-1}
    alpha: float  # smoothing weight in (0, 1]; higher = more weight on recent data
    # default alpha used by the monitor is 0.2, following ECDD's recommended lambda (Lu et al. 2019)
) -> float:
    
    # return updated EWMA value:
    return alpha * current_value + (1.0 - alpha) * previous_ewma


# calculates Wasserstein distance between two violation-score distributions
# measures the minimum "cost" to transform one distribution into the other, capturing changes in distributional shape between 
# consecutive batches:
def compute_divergence(
    violations_current: np.ndarray,  # violation scores from the current batch
    violations_previous: np.ndarray,  # violation scores from the previous batch
) -> float:

    # return Wasserstein-1 distance between the two empirical distributions:
    return float(wasserstein_distance(violations_current, violations_previous))


# control-limit alert
# fire when the current (smoothed) mean violation exceeds the baseline mean by more than 'threshold' baseline standard deviations: 
def check_alert(
    current_mean: float,  # mean violation for the current batch (or its EWMA)
    baseline_mean: float,  # mean violation over the training data (reference)
    baseline_std: float,  # std of violations over the training data
    threshold: float,  # number of baseline std devs above baseline before alerting
                       # (the control-limit multiplier; ECDD's "L", Lu et al. 2019)
) -> bool:

    # if baseline violations are essentially constant, any rise above the mean is a meaningful 
    # deviation (avoid a degenerate 0 * std control limit):
    if baseline_std < 1e-8:
        return current_mean > baseline_mean
    return current_mean > baseline_mean + threshold * baseline_std # True if an alert should be raised