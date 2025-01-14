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


def dirichlet_mean(labels, prior):
    classes = np.array(list(range(len(prior))))
    counts = np.zeros(len(classes), dtype=int)

    for i, cls in enumerate(classes):
        counts[i] = np.sum(labels == cls)

    post = prior + counts
    return 1 - post.max() / post.sum()



def get_Beta(xs, models, threshold=0.5, prior_a=0.5, prior_b=0.5):
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


def get_dirichlet(xs, models, prior=None):
    if prior is None:
        prior = np.array([0.5]*_check_boundary_response_method(models[0], "auto")(xs[0]).shape[0])
    preds = np.array([_check_boundary_response_method(m, "auto")(xs) for m in models])
    # preds are probabilities

    labels = np.argmax(preds, axis=2)

    return np.array([dirichlet_mean(label, prior) for label in labels.T])
