"""
Classifiers for generating the predictions used to compute ground-truth fairness metrics (DI, AOD, BalAcc).

Note that these models exist only to produce predicted labels, from which the ground-truth fairness metrics are computed. 
The CC-based monitor is model-agnostic and does not depend on them: it tracks distributional drift in the feature space independently 
of any classifier. The models are what the monitor is validated AGAINST, not part of the monitoring mechanism.

Two models, matching ConFair (Yang & Meliou, 2024, TrainMLModels.py):
  - LR  : sklearn LogisticRegression (their 'lr')
  - XGB : xgboost XGBClassifier (their 'tr')

ConFair tunes XGB via grid search over a validation set (n_estimators in {5,10}, max_depth in {2,3,5}, learning_rate in
{0.001,0.01,0.1,0.2,0.3}). Because this project uses a 70/30 split with NO validation set (see bundle.py), grid search is not possible. 
Instead a fixed configuration from within ConFair's grid range is used (n_estimators=10, max_depth=3, learning_rate=0.1). 
Consequently XGB baseline numbers may differ from ConFair's grid-searched values more than the LR numbers do; this is the documented 
tradeoff of the no-validation-set design.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# fixed XGB config drawn from within ConFair's grid-search range:
XGB_PARAMS = dict(
    n_estimators=10,
    max_depth=3,
    learning_rate=0.1,
)

# train and return a fitted classifier:
def train_model(
    model_name: str,  # 'lr' or 'xgb'
    X_train: np.ndarray,  # (n, d) training features
    y_train: np.ndarray,  # (n,) training labels
    seed: int = 42,  # random seed for reproducibility
):

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)

    if model_name == "lr":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "xgb":
        model = XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            **XGB_PARAMS,
        )
    else:
        raise ValueError(f"Unknown model_name {model_name!r}; choose 'lr' or 'xgb'.")

    model.fit(X_train, y_train)
    return model  # a fitted sklearn-style classifier with a .predict() method

# return binary predictions (0/1) for X:
def predict(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return model.predict(X).astype(int)