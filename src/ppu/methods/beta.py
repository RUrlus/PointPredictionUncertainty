import numpy as np
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method


def beta_mean(lables, prior_a=0.5, prior_b=0.5):
    m = np.count_nonzero(lables)
    n = len(lables)
    p = (prior_a + m) / (prior_a + prior_b + n)
    return min(p, 1 - p)


def beta_para(lables, prior_a=0.5, prior_b=0.5):
    m = np.count_nonzero(lables)
    n = len(lables)
    alpha = prior_a + m
    beta = n - m + prior_b
    return alpha, beta


def get_Beta(xs, models, threshold, prior_a=0.5, prior_b=0.5):
    preds = np.array([_check_boundary_response_method(m, "auto")(xs) for m in models])
    # preds are probabilities
    if len(preds.shape) == 3:
        preds = preds[:, :, 1]

    labels = np.where(preds >= threshold, 1, 0)

    return np.array([beta_mean(label, prior_a, prior_b) for label in labels.T])


def get_Beta_para(start, end, models, threshold, prior_a=0.5, prior_b=0.5, num_points=200):
    x_coords = np.linspace(start[0], end[0], num_points)
    y_coords = np.linspace(start[1], end[1], num_points)
    points = np.vstack((x_coords, y_coords)).T
    preds = np.array([_check_boundary_response_method(m, "auto")(points) for m in models])
    # preds are probabilities
    if len(preds.shape) == 3:
        preds = preds[:, :, 1]

    labels = np.where(preds >= threshold, 1, 0)
    return np.array([beta_para(label, prior_a, prior_b) for label in labels.T])
