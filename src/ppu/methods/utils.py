import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.datasets import make_moons


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
    r""" 
    LSE lower bound estimator
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
    sums = np.sum([np.log1p(avg_sq / (4*np.pi**2*(j+0.5)**2)) for j in range(C)])
    return avg/2 + sums + np.log(2)

def BI_LSE(z, bound="auto", C=1000):
    r"""Bregman information generated by LSE(x) = ln(1 + \sum_i e^{x_i})
    `bound`={'auto', 'lower', 'upper'} sets if upper or lower bound should be estimated;
    'upper' is only possible for the binary setting;
    'auto' selects 'upper' for binary and 'lower' for non-binary data;
    input: reduced logits
    """
    if len(z.shape) < 2:
        z = np.expand_dims(z, axis=1)

    if bound=="auto":
        # avg = np.mean(z)
        # xxT = np.outer(z, z)
        # avg_sq = np.mean(xxT)*n/(n-1) - np.mean(np.diag(xxT))/(n-1)
        # # if False, we would get log of a negative value
        # log_of_pos = 0 < (1 + (np.mean(z)**2 - np.var(z)/n)/(16*np.pi**2))
        log_of_pos = True
        upper_valid = (z.shape[1]==1) and log_of_pos
        bound = "upper" if upper_valid else "lower"

    E_of_LSE = np.mean(LSE(z))

    if bound=="upper":
        LSE_of_E = LSE_LB_estimator_(z, C)
    elif bound=="lower":
        avg_z = np.mean(z, axis=0)
        LSE_of_E = LSE(np.expand_dims(avg_z, axis=0))[0]
    else:
        raise NotImplementedError

    return E_of_LSE - LSE_of_E

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

def generate_data(n, seed, shape="circular", noise=0.5):

    np.random.seed(seed)
    var = noise

    assert n % 2 == 0

    if shape == "circular":
        # sample polar coordinates
        angles = np.random.uniform(low=0, high=2*np.pi, size=n)
        radii = ys = np.random.binomial(n=1, p=0.5, size=n)
        # transform to cartesian coordinates and add noise
        x1 = np.sin(angles)*radii + np.random.normal(scale=var, size=n)
        x2 = np.cos(angles)*radii + np.random.normal(scale=var, size=n)

    elif shape == "binormal":
        ys = np.random.binomial(n=1, p=0.5, size=n)
        mu_1 = 0.5 - ys
        mu_2 = ys - 0.5
        x1 = np.random.normal(loc=mu_1, scale=var, size=n)
        x2 = np.random.normal(loc=mu_2, scale=var, size=n)

    elif shape == "moon":
        pass

    xs = np.array([x1, x2]).T
    return xs, ys

def get_datasets(seed, n_samples=100, n_test_samples=200):
    moon_set = (
        make_moons(n_samples=n_samples, noise=0.3, random_state=seed),
        make_moons(n_samples=n_test_samples, noise=0.3, random_state=seed+1000)
    )
    circular_set = (
        generate_data(n=n_samples, shape="circular", seed=seed, noise=0.3),
        generate_data(n=n_test_samples, shape="circular", seed=seed+1000, noise=0.3),
    )
    binormal_set = (
        generate_data(n=n_samples, shape="binormal", seed=seed, noise=0.6),
        generate_data(n=n_test_samples, shape="binormal", seed=seed+1000, noise=0.6),
    )
    return moon_set, circular_set, binormal_set


def read_results(severity=5, unc_type="Conf", ds_name="Cifar10-C", target="Accuracy"):
    results_ = pd.read_pickle(f"results/{ds_name}/{target}/{unc_type}_sev{severity}_all.pkl")
    results_ = results_[results_["Accuracy"].notnull()]

    zipped_ = zip(results_[target], results_["Classif perc"])
    results_[target] = [v + w*1j for v, w in zipped_]

    if unc_type == "Conf":
        results_["Conf Quantile"] = 1 - results_["Conf Quantile"]
        results_["Uncertainty"] = "Confidence Score"

    elif unc_type == "DE":
        results_["Uncertainty"] = "BI DE (ours)"

    elif unc_type == "BS":
        results_["Uncertainty"] = "BI BS (ours)"

    elif unc_type == "TTA":
        results_["Uncertainty"] = "BI TTA (ours)"

    elif unc_type == "MCD":
        results_["Uncertainty"] = "BI MCDropout (ours)"

    results_["Corruption Severity"] = results_["Corruption"].replace({
        "None": "0",
        "brightness": str(severity),
        "fog": str(severity),
        "glass_blur": str(severity),
        "pixelate": str(severity),
        "spatter": str(severity),
        "contrast": str(severity),
        "frost": str(severity),
        "impulse_noise": str(severity),
        "saturate": str(severity),
        "speckle_noise": str(severity),
        "defocus_blur": str(severity),
        "gaussian_blur": str(severity),
        "jpeg_compression": str(severity),
        "shot_noise": str(severity),
        "zoom_blur": str(severity),
        "elastic_transform": str(severity),
        "gaussian_noise": str(severity),
        "motion_blur": str(severity),
        "snow": str(severity)
    })
    results_.rename(columns={"Conf Quantile": "Validation set quantile", "BI Quantile": "Validation set quantile"}, inplace=True)

    return results_.reset_index()
