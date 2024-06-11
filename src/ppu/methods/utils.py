import random
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

"""
The following 6 functions are adapted from the following source:

Source: https://github.com/MLO-lab/Uncertainty_Estimates_via_BVD.git
Author: SebGGruber
License: MIT License
"""
def weighted_mean(x, **kws):
    return np.sum(np.real(x) * np.imag(x)) / np.sum(np.imag(x))

def weighted_sd(x, **kws):
    mu = weighted_mean(x)
    sd = np.sqrt(np.sum((np.real(x) - mu)**2 * np.imag(x)) / np.sum(np.imag(x)))
    return mu-sd, mu+sd

def stable_logit_transform(probs, max_=16):
    with np.errstate(divide = "ignore", invalid="ignore"):
        probs = np.array(probs)
        logs = np.log(probs)
        anti_logs = np.log(np.subtract(1, probs))
        results = logs - anti_logs
        return np.clip(results, -max_, max_)

def LSE(z):
    """input: reduced logits"""
    #if len(x.shape) < 2:
    z = np.hstack([z, np.zeros(z.shape)])
    #x[:,1] = 0
    return logsumexp(z, axis=1)

def LSE_LB_estimator_(x, C):
    r"""LSE lower bound estimator
    C: number of summands in the estimator - higher is better but slower

    Computes the following formula:
    \frac{1}{2n} \\sum_{i=1}^n X_i + \\sum_{j=1}^C \\ln (1 + \frac{\frac{1}{n(n-1)} \\sum_{k \neq l} X_k X_l}{4 (n - 0.5)^2 \\pi^2}) + \\ln 2
    """
    assert x.shape[1]==1
    n = x.shape[0]
    avg = np.mean(x)
    xxT = np.outer(x, x)
    # \frac{1}{n(n-1)} \sum_{i \neq j} X_i X_j
    avg_sq = np.mean(xxT)*n/(n-1) - np.mean(np.diag(xxT))/(n-1)
    #avg_sq = avg**2 - np.var(x)/n
    sums = np.log1p(avg_sq / (4 * np.pi**2 * np.arange(0.5, C + 0.5) ** 2)).sum()
    return avg/2 + sums + np.log(2)

def accuracy(predictions, labels):
    # array indicating which prediction is correct
    correct = np.equal(predictions, labels)
    # number of positives divided by number overall
    return np.sum(correct)/len(correct)

def get_dataset(rng, gen, n_samples=200, n_test_samples=200, **kwargs):
    gen = gen(rng=rng, **kwargs)
    return (
        gen.rvs(n_samples),
        gen.rvs(n_test_samples)
    )

def get_models(clf, gen, reps, n_samples=200, **kwargs):
    result = []
    for rng in range(reps):
        (X_train, y_train), (X_test, y_test) = get_dataset(rng, gen, n_samples=n_samples, **kwargs)
        new_clf = deepcopy(clf)
        new_clf.fit(X_train, y_train)
        result.append(new_clf)
    return result

def bootstrap_models(clf, gen, reps, n_samples=500, **kwargs):
    result = []
    seeds = range(reps)
    (X_train, y_train), (X_test, y_test) = get_dataset(0, gen, n_samples=n_samples, **kwargs)
    idx = np.arange(n_samples)
    for _ in seeds:
        bs_ind = random.choices(idx, k=n_samples)
        bs_X = X_train[bs_ind]
        bs_y = y_train[bs_ind]
        model = deepcopy(clf)
        model.fit(bs_X, bs_y)
        result.append(model)
    return result
