"""
DatasetBundle: standardized container for preprocessed dataset splits.

This is the interface between the preprocessing pipeline & the CC monitoring experiments. All dataset loaders produce a 
DatasetBundle, & all downstream code (CC learning, drift simulation, monitoring) consumes one.
DatasetBundle is a standardized container for a preprocessed dataset split into train & test sets, w/ feature matrix, labels, & 
protected attribute values.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


# The following is a shared contract btwn the preprocessing pipeline & the experiment code. 
# instead of passing X_train, y_train, protected_train, etc. as separate arguments everywhere, I package them into one object.
@dataclass
class DatasetBundle:
    X_train: np.ndarray  # training feature matrix, shape (n_train, n_features)
    X_test: np.ndarray  # test feature matrix, shape (n_test, n_features)
    y_train: np.ndarray  # training labels (binary: 0 or 1), shape (n_train,)
    y_test: np.ndarray  # test labels (binary: 0 or 1), shape (n_test,)
    protected_train: np.ndarray  # protected attribute values for training set (binary: 0 = minority/protected group, 1 = majority/privileged group), shape (n_train,)
    protected_test: np.ndarray  # protected attribute values for test set (binary: 0 or 1), shape (n_test,)
    dataset_name: str  # name of dataset (e.g. "LSAC", "MEPS", "Credit")
    feature_names: List[str]  # names of the feature columns, in same order as columns in X_train & X_test
    protected_name: str  # name of protected attribute column (e.g. "race", "RACE", "age")
    label_name: str  # name of label column (e.g. "pass_bar", "UTILIZATION", "SeriousDlqin2yrs")
    continuous_indices: List[int]  # Column indices of X_train/X_test that are continuous (> 8 distinct values)
        # Note: This is following ConFair's num_threshold=8 rule. Conformance Constraints are built on continuous features ONLY 
        # (Fariha et al. 2021 Algorithm 1 line 1 drops non-numerical attributes; ConFair's LearnCCrules.py builds CCs only on the
        # continuous columns). So, the CC code reads this field to select the right columns

'''
Note on lack of validation set:

ConFair (Yang & Meliou, 2024) uses a 70/15/15 train/val/test split, with the validation set used for model hyperparameter tuning 
via grid search. I deliberately deviate from this. 

I do that because the core contribution of this thesis is the correlation between CC violation signals and fairness + utility metrics 
(DI, AOD, BalAcc) computed batch-by-batch over a temporal drift stream. The test set is the pool from which drift batches are sampled,
and with 20 batches per experiment, a 15% test set yields ~117-183 rows per batch (depending on dataset), which would be more likely
to produce noisy per-batch fairness metric estimates and weaken the correlation signal.

A 30% test set approximately doubles batch sizes, improving the statistical reliability of the fairness metrics that are the 
dependent variable in my correlation analysis.

The tradeoff is that my models are not hyperparameter-tuned the same way as Yang and Meliou (2024). This means my baseline fairness 
numbers (NO-INTERVENTION) may differ slightly from theirs. I will note this difference explicitly when making comparisons.
'''