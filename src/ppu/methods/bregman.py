"""Module with Bregman Information implementation.

Based on:
Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition
Sebastian G. Gruber and Florian Buettner
"""

import numpy as np
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method

from ppu.methods.utils import BI_LSE, stable_logit_transform


def get_BI(xs, models):
    preds = np.array([_check_boundary_response_method(m, "predict_proba")(xs) for m in models])
    # preds are probabilities
    if len(preds.shape) == 3:
        preds = preds[:,:,1]

    logits = stable_logit_transform(preds)

    BIs = np.array([BI_LSE(zs, bound="lower") for zs in logits.T])
    return BIs

def get_revised_BI(xs, models):
    preds = np.array([_check_boundary_response_method(m, "auto")(xs) for m in models])
    # preds are probabilities
    if len(preds.shape) == 3:
        preds = preds[:,:,1]

    logits = stable_logit_transform(preds)

    BIs = np.array([BI_LSE(zs, bound="lower") / abs(b.mean() - 0.5) for b,zs in zip(preds.T, logits.T)])
    # BIs = np.array([BI_LSE(zs, bound="lower") / (-0.05 / (abs(b.mean() - 0.5) + 0.07) + 1) for b,zs in zip(preds.T, logits.T)])
    # BIs = np.array(softmax([10*(1 - norm.cdf(abs(b.mean() - 0.5) / b.var()))-1. for b,zs in zip(preds.T, logits.T)]))
    return BIs


