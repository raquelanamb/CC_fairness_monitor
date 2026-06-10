"""
Credit (Give Me Some Credit) dataset loader.
 
Reads the preprocessed CSV from data/processed/credit.csv and returns a DatasetBundle with train/test splits.

Split method follows ConFair's (Yang & Meliou, 2024) PrepareData.py: np.random.seed(seed) + np.random.permutation() rather 
than sklearn's train_test_split, for reproducibility and comparability with their pipeline.

Split ratio: 70/30 train/test.
See bundle.py for the rationale behind using 70/30 instead of ConFair's 70/15/15.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sys import path as sys_path

# allow import of DatasetBundle from parent directory:
sys_path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle import DatasetBundle


PROCESSED_PATH = Path("data/processed/credit.csv")

LABEL_COL = "SeriousDlqin2yrs"
PROTECTED_COL = "age"
DATASET_NAME = "Credit"

# ConFair seeds used for their 20-run evaluation:
CONFAIR_SEEDS = [
    1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121,
    50, 583, 5278, 100000, 48879, 51966, 57005, 7777, 100, 923
] # (used later in the experiment pipeline, when I run a 20-run baseline comparison against ConFair's NO-INTERVENTION results)

# columns w/ more than this many distinct values are treated as continuous (ConFair's num_threshold=8 rule):
NUM_THRESHOLD = 8

# loads the preprocessed Credit dataset & returns a DatasetBundle:
def load_credit(
    processed_path: Path = PROCESSED_PATH,
    seed: int = 42,
    train_size: float = 0.7,
) -> DatasetBundle:

    # read processed path into df:
    df = pd.read_csv(processed_path)

    # separate features, label, & protected attribute:
    feature_names = [c for c in df.columns if c not in [LABEL_COL, PROTECTED_COL]]

    # convert pandas df columns into numpy arrays:
    X = df[feature_names].values.astype(float)
    y = df[LABEL_COL].values.astype(int)
    protected = df[PROTECTED_COL].values.astype(int)

    # identify continuous feature columns (> NUM_THRESHOLD distinct values), per ConFair's num_threshold=8 rule.
    # CCs are built on continuous features ONLY (Fariha et al. Alg. 1 line 1; ConFair LearnCCrules.py builds CCs only on the 
    # continuous columns):
    continuous_indices = [
        i for i in range(X.shape[1])
        if len(np.unique(X[:, i])) >= NUM_THRESHOLD
    ]

    # split following ConFair's method (np.random.seed + np.random.permutation, not sklearn train_test_split):
    n = len(df)
    np.random.seed(seed) # sets random seed so shuffle is reproducible
    order = np.random.permutation(n) # creates randomly shuffled array of indices from 0 to n-1
    split_point = int(train_size * n)

    train_idx = order[:split_point] # first 70% of shuffled indices = training indices
    test_idx  = order[split_point:] # remaining 30% = test indices

    return DatasetBundle(
        X_train=X[train_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
        protected_train=protected[train_idx],
        protected_test=protected[test_idx],
        dataset_name=DATASET_NAME,
        feature_names=feature_names,
        protected_name=PROTECTED_COL,
        label_name=LABEL_COL,
        continuous_indices=continuous_indices,
    )