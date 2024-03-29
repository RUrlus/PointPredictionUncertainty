from ppu.generator import GaussianBlobs, Circular, Moons, RingBlobs
from copy import deepcopy


def get_dataset(rng, gen, n_samples=200, n_test_samples=200, **kwargs):
    gen = gen(rng=rng, **kwargs)
    dataset = (
        gen.rvs(n_samples),
        gen.rvs(n_test_samples)
    )
    return dataset


def get_models(clf, gen, reps, n_samples=200):
    result = []
    for rng in range(reps):
        (X_train, y_train), (X_test, y_test) = get_dataset(rng, gen, n_samples=n_samples)
        new_clf = deepcopy(clf)
        new_clf.fit(X_train, y_train)
        result.append(new_clf)
    return result


def BS_nn_models(gen, n_models, n_samples=500, **kwargs):
    # bootstrapping
    result = []
    seeds = range(n_models)
    
    (X_train, y_train), (X_test, y_test) = get_dataset(0, gen, n_samples=n_samples)
    for _ in seeds:
        bs_ind = random.choices(range(n_samples), k=n_samples)
        bs_X = X_train[bs_ind]
        bs_y = y_train[bs_ind]
        model = MLP(**kwargs)
        model.fit(bs_X, bs_y)
        result.append(model)
    return result


def BS_models(clf, gen, reps, n_samples=500):
    result = []
    seeds = range(reps)
    (X_train, y_train), (X_test, y_test) = get_dataset(0, gen, n_samples=n_samples)
    for _ in seeds:
        bs_ind = random.choices(range(n_samples), k=n_samples)
        bs_X = X_train[bs_ind]
        bs_y = y_train[bs_ind]
        model = clf
        model.fit(bs_X, bs_y)
        result.append(model)
    return result

