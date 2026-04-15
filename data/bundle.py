"""
DatasetBundle: standardized container for preprocessed dataset splits.

This is the interface between the preprocessing pipeline & the CC monitoring experiments. All dataset loaders produce a 
DatasetBundle, & all downstream code (CC learning, drift simulation, monitoring) consumes one.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


# DatasetBundle is a shared contract btwn the preprocessing pipeline & the experiment code. 
# instead of passing X_train, y_train, protected_train, etc. as separate arguments everywhere, I package them into one object.
@dataclass
class DatasetBundle:
    """
    Standardized container for a preprocessed dataset split into train & test sets, w/ feature matrix, labels, & protected 
    attribute values.

    Fields
    ------
    X_train : np.ndarray, shape (n_train, n_features)
        Training feature matrix
    X_test : np.ndarray, shape (n_test, n_features)
        Test feature matrix
    y_train : np.ndarray, shape (n_train,)
        Training labels (binary: 0 or 1)
    y_test : np.ndarray, shape (n_test,)
        Test labels (binary: 0 or 1)
    protected_train : np.ndarray, shape (n_train,)
        Protected attribute values for training set (binary: 0 = minority/protected group, 1 = majority/privileged group)
    protected_test : np.ndarray, shape (n_test,)
        Protected attribute values for test set (binary: 0 or 1)
    dataset_name : str
        Human-readable name of the dataset (e.g. "LSAC", "MEPS", "Credit")
    feature_names : List[str]
        Names of the feature columns, in the same order as columns in X_train and X_test
    protected_name : str
        Name of the protected attribute column (e.g. "race", "RACE", "age")
    label_name : str
        Name of the label column (e.g. "pass_bar", "UTILIZATION", "SeriousDlqin2yrs")
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    protected_train: np.ndarray
    protected_test: np.ndarray
    dataset_name: str
    feature_names: List[str]
    protected_name: str
    label_name: str


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