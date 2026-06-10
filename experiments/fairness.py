"""
Fairness and utility metrics: Disparate Impact (DI), Average Odds Difference (AOD), and Balanced Accuracy (BalAcc).

These replicate ConFair's exact computation (Yang & Meliou, 2024, EvaluateModels.py: eval_predictions / eval_settings) rather than calling
AIF360, so that the NO-INTERVENTION baseline numbers can be compared directly against ConFair's reported values. AIF360's metric 
conventions (which group is the numerator, sign conventions) differ subtly and would not line up.

Group coding (matching the DatasetBundle / ConFair):
  G0 = protected == 0 = minority / protected group
  G1 = protected == 1 = majority / privileged group

Definitions (from ConFair EvaluateModels.py):
  Per group:  TPR = TP/P, TNR = TN/N, FPR = FP/N, SR = predicted-positive rate
  DI = SR_G0 / SR_G1  (0 if SR_G1 == 0)
  AOD = 0.5 * [(FPR_G0 - FPR_G1) + (TPR_G0 - TPR_G1)]
  BalAcc = 0.5 * (TPR + TNR) over the WHOLE test set (not per group)

Transformed variants (for consistent "higher = fairer" orientation in the correlation analysis):
  DI*  = min(DI, 1/DI)   (1 = parity; <1 = disparity in either direction)
  AOD* = 1 - |AOD|       (1 = parity; ->0 = disparity)
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# 0.5 * (TPR + TNR) - shared by the threshold tuner (models.py) and compute_metrics:
def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    TPR = TP / P if P > 0 else 0.0
    TNR = TN / N if N > 0 else 0.0
    return 0.5 * (TPR + TNR)


# compute TPR, TNR, FPR, and SR (selection rate) for one group, following ConFair's eval_predictions (confusion_matrix with labels=[0,1]):
def _group_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    P = TP + FN  # actual positives
    N = TN + FP  # actual negatives
    total = TP + TN + FP + FN

    TPR = TP / P if P > 0 else 0.0
    TNR = TN / N if N > 0 else 0.0
    FPR = FP / N if N > 0 else 0.0
    SR = (TP + FP) / total if total > 0 else 0.0  # predicted-positive rate

    return {"TPR": TPR, "TNR": TNR, "FPR": FPR, "SR": SR}


# compute DI, AOD, BalAcc (and their transformed variants) for one set of predictions, replicating ConFair's eval_settings:
def compute_metrics(
    y_true: np.ndarray,  # (n,) ground-truth binary labels.
    y_pred: np.ndarray,  # (n,) predicted binary labels.
    protected: np.ndarray,  # (n,) protected attribute (0 = minority/G0, 1 = majority/G1)
) -> dict:

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    protected = np.asarray(protected, dtype=int)

    g0 = protected == 0  # minority
    g1 = protected == 1  # majority

    r0 = _group_rates(y_true[g0], y_pred[g0])
    r1 = _group_rates(y_true[g1], y_pred[g1])

    # DI = SR_G0 / SR_G1  (ConFair: 0 if SR_G1 == 0)
    DI = r0["SR"] / r1["SR"] if r1["SR"] > 0 else 0.0

    # AOD = 0.5 * [(FPR_G0 - FPR_G1) + (TPR_G0 - TPR_G1)]
    AOD = 0.5 * ((r0["FPR"] - r1["FPR"]) + (r0["TPR"] - r1["TPR"]))

    # BalAcc over the WHOLE set (ConFair: bal_acc_all)
    BalAcc = balanced_accuracy(y_true, y_pred)

    # transformed variants (higher = fairer, bounded in [0, 1]):
    DI_star = min(DI, 1.0 / DI) if DI > 0 else 0.0
    AOD_star = 1.0 - abs(AOD)

    # return dict with keys: DI, AOD, BalAcc, DI_star, AOD_star, plus per-group rates for transparency:
    return {
        "DI": DI,
        "AOD": AOD,
        "BalAcc": BalAcc,
        "DI_star": DI_star,
        "AOD_star": AOD_star,
        "G0": r0,
        "G1": r1,
    }