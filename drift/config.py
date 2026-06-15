"""
Configuration dataclasses for temporal drift simulation (thesis Objective 3: "Design a temporal data stream simulation framework from 
static fairness datasets by introducing controlled group-based and covariate drifts").

Drift types:
  - GROUP_COVARIATE: shift features for ONE subgroup (the fairness-critical case; asymmetric drift that harms one group)
  - GLOBAL_COVARIATE: shift features for ALL points (broad covariate shift)

Temporal patterns (map to the drift trends described in Methodology 3.4):
  - GRADUAL: linear ramp of magnitude from onset to the final batch
  - ABRUPT: step change to full magnitude at the onset batch
  - VARIANCE: increasing spread (scale) of features, ramping from 1.0
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

import numpy as np


class DriftType(Enum):
    GROUP_COVARIATE = "group_covariate"  # shift features for ONE subgroup
    GLOBAL_COVARIATE = "global_covariate"  # shift features for ALL points


class DriftPattern(Enum):
    GRADUAL = "gradual"  # linear ramp-up of drift magnitude
    ABRUPT = "abrupt"  # step function at onset batch
    VARIANCE = "variance"  # increasing spread (scale) of features


# specification for a single drift injection:
@dataclass
class DriftSpec:
    drift_type: DriftType  # whether drift affects one group or all points
    drift_pattern: DriftPattern  # temporal shape (gradual / abrupt / variance)
    onset_batch: int  # batch index where drift begins (0-indexed). batch 0 is always drift-free (the deployment baseline)
    magnitude: float  # drift strength, in units of training-set std dev
        # - GRADUAL: magnitude reached at the final batch
        # - ABRUPT: constant magnitude after onset
        # - VARIANCE: scale multiplier at final batch (1.0 = no change, 2.0 = doubled std dev)
    affected_group: Optional[int] = None  # protected value to target. Required for GROUP_COVARIATE, ignored for GLOBAL_COVARIATE
    feature_indices: Optional[List[int]] = None  # which feature columns to drift. None = all features
        # - Note: the CC monitor only sees continuous features, so to produce a signal the drifted features should overlap the bundle's 
        #   continuous_indices (the runner handles this).
    random_state: Optional[int] = 42  # seed for reproducibility


# a single temporal batch plus ground-truth drift metadata:
@dataclass
class BatchResult:
    batch_index: int
    X: np.ndarray
    y: np.ndarray
    protected: np.ndarray
    drift_applied: bool
    drift_type: Optional[DriftType] = None
    drift_pattern: Optional[DriftPattern] = None
    drift_magnitude_actual: float = 0.0
    affected_group: Optional[int] = None

    @property
    def n_samples(self) -> int: 
        return len(self.X)


# complete result of a temporal stream simulation:
@dataclass
class StreamResult:
    batches: List[BatchResult]
    drift_spec: DriftSpec
    dataset_name: str
    n_batches: int
    batch_size: int
    feature_stds: np.ndarray
    feature_means: np.ndarray