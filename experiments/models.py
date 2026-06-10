"""
Classifiers for generating the predictions used to compute ground-truth fairness metrics (DI, AOD, BalAcc).

Note that these models exist only to produce predicted labels, from which the ground-truth fairness metrics are computed. 
The CC-based monitor is model-agnostic and does not depend on them: it tracks distributional drift in the feature space independently 
of any classifier. The models are what the monitor is validated AGAINST, not part of the monitoring mechanism.

Two models, matching ConFair (Yang & Meliou, 2024, TrainMLModels.py):
  - LR: sklearn LogisticRegression (their 'lr'), wrapped in a StandardScaler pipeline so the optimizer converges and features are 
        scaled as in ConFair
  - XGB: xgboost XGBClassifier (their 'tr')

ConFair tunes XGB via grid search over a validation set (n_estimators in {5,10}, max_depth in {2,3,5}, learning_rate in
{0.001,0.01,0.1,0.2,0.3}). Because this project uses a 70/30 split with NO validation set (see bundle.py), grid search is not possible. 
Instead a fixed configuration from within ConFair's grid range is used (n_estimators=10, max_depth=3, learning_rate=0.1).
 
Decision threshold: ConFair does not predict at the default 0.5 cutoff. It scans 100 thresholds in [0.01, 0.99] and selects the one 
maximizing balanced accuracy (find_optimal_thres, opt_obj='BalAcc'), then applies that threshold to the predicted probabilities. I 
replicate this. Because this project has no validation set, the threshold is selected on the TRAINING set (ConFair uses a validation set); 
this is a documented consequence of the no-validation design.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from experiments.fairness import balanced_accuracy


XGB_PARAMS = dict(n_estimators=10, max_depth=3, learning_rate=0.1)


# scan 100 thresholds in [0.01, 0.99], return the one maximizing balanced
# accuracy (first if tied), replicating ConFair's find_optimal_thres:
def find_optimal_thres(y_true, y_scores, num_thresh: int = 100) -> float:
    thresholds = np.linspace(0.01, 0.99, num_thresh)
    ba = np.zeros(num_thresh)
    for i, t in enumerate(thresholds):
        ba[i] = balanced_accuracy(y_true, (y_scores > t).astype(int))
    best_ind = np.where(ba == ba.max())[0][0]   # first max, matching ConFair
    return float(thresholds[best_ind])


# train a classifier and select a BalAcc-optimal threshold on training data:
def train_model(model_name, X_train, y_train, seed=42, tune_threshold=True):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    if model_name == "lr":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
    elif model_name == "xgb":
        model = XGBClassifier(random_state=seed, eval_metric="logloss", **XGB_PARAMS)
    else:
        raise ValueError(f"Unknown model_name {model_name!r}; choose 'lr' or 'xgb'.")

    model.fit(X_train, y_train)

    if tune_threshold:
        train_scores = model.predict_proba(X_train)[:, 1]
        threshold = find_optimal_thres(y_train, train_scores)
    else:
        threshold = 0.5

    return model, threshold


# return binary predictions (0/1) using the given probability threshold:
def predict(model, X, threshold: float = 0.5) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    scores = model.predict_proba(X)[:, 1]
    return (scores > threshold).astype(int)