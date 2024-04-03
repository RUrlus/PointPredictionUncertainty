"""Module with Bregman Information implementation.

Based on:
Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition
Sebastian G. Gruber and Florian Buettner
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

from matplotlib.colors import ListedColormap
from ppu.methods.utils import BI_LSE, accuracy, stable_logit_transform, get_dataset
from ppu.methods.mlp import MLP
from copy import deepcopy
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from scipy.special import expit
from sklearn.inspection import DecisionBoundaryDisplay




def get_BI(xs, models):
    preds = np.array([_check_boundary_response_method(m, 'auto')(xs) for m in models])
    # preds might be probabilities
    if len(preds.shape) == 3:
        logits = stable_logit_transform(preds[:,:,1])
    # or just logits
    else:
        logits = preds
        
    BIs = np.array([BI_LSE(zs, bound='lower') for zs in logits.T])
    return BIs

def get_models(clf, gen, reps, n_samples=200, **kwargs):
    result = []
    for rng in range(reps):
        (X_train, y_train), (X_test, y_test) = get_dataset(rng, gen, n_samples=n_samples, **kwargs)
        new_clf = deepcopy(clf)
        new_clf.fit(X_train, y_train)
        result.append(new_clf)
    return result

def nn_models(gen, n_models, n_samples=500, DE=False, extra_kwargs={}, **kwargs):
    result = []
    # if we want deep ensembles, we have to fix the dataset seed to get the same dataset
    seeds = [0 for _ in range(n_models)] if DE else range(n_models)
    
    for rng in seeds:
        (X_train, y_train), (X_test, y_test) = get_dataset(rng, gen, n_samples=n_samples, **extra_kwargs)
        model = MLP(**kwargs)
        model.fit(X_train, y_train)
        result.append(model)
    return result

def BS_models(clf, gen, reps, n_samples=500, **kwargs):
    result = []
    seeds = range(reps)
    (X_train, y_train), (X_test, y_test) = get_dataset(0, gen, n_samples=n_samples, **kwargs)
    for _ in seeds:
        bs_ind = random.choices(range(n_samples), k=n_samples)
        bs_X = X_train[bs_ind]
        bs_y = y_train[bs_ind]
        model = deepcopy(clf)
        model.fit(bs_X, bs_y)
        result.append(model)
    return result

def BS_nn_models(gen, n_models, n_samples=500, DE=False, extra_kwargs={}, **kwargs):

    # bootstrapping
    result = []
    seeds = range(n_models)
    
    (X_train, y_train), (X_test, y_test) = get_dataset(0, gen, n_samples=n_samples, **extra_kwargs)
    for _ in seeds:
        bs_ind = random.choices(range(n_samples), k=n_samples)
        bs_X = X_train[bs_ind]
        bs_y = y_train[bs_ind]
        model = MLP(**kwargs)
        model.fit(bs_X, bs_y)
        result.append(model)
    return result


def diff_classifier(generator, n_samples, names, classifiers, models, rescale = False):
    dataset = get_dataset(1000, generator, n_samples=n_samples)
    offset = 1
    rows = 2

    gridspec_kw={'height_ratios': [1, 1]}
    cbar_kws = dict(use_gridspec=False, location="bottom")
    plt.rcParams.update({'font.size': 15})
    figure, axs = plt.subplots(rows, len(models) + offset, gridspec_kw=gridspec_kw, figsize=(27, 6))

    i = 0
    eps = 1
    n_ticks = 100

    (X_train, y_train), (X_test, y_test) = dataset  # X is the data point, y is the class label


    # the area we draw is a bit larger than the range of data points
    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

    # just plot the dataset first
    cm = plt.cm.RdBu # color map parameter
    cm_bright = ListedColormap(["#FF0000", "#0000FF"]) # color for data points 
    ax = axs[0][i] # position of subgraph
    ax.set_title("Data") # subgraph title
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.set_xlim(x_min, x_max) # axis range
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, clf in zip(names, classifiers):
        ax = axs[0][i]
        clf.fit(X_train, y_train) # clf is the method in classifiers
        score = clf.score(X_test, y_test) # score of the result
        x = np.linspace(x_min, x_max, n_ticks) # for the heatmap we need to creat the grid first, so n_ticks are the density of the grid nodes
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()] # All the grid points

        if name == 'Neural Net':
            response = clf.predict(X_grid)
        else: # this if else is no needed
            response = _check_boundary_response_method(clf, 'auto')(X_grid) # the function returns the probability of points(inputs) belong to each class

        if len(response.shape) == 1:
            response = expit(response)
        else:
            response = response[:, 1] # since there's only 2 classes so 1 can represent the other(the prob sums up to 1) 

        display = DecisionBoundaryDisplay(
            xx0=xs,
            xx1=ys,
            response=response.reshape(xs.shape),
        ) # class to draw DecisionBoundary
        display.plot(ax=ax, cmap=cm, alpha=0.8) # alpha controls the transparency of the filled contours

        ax.set_xticks(()) # remove the ticks
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            horizontalalignment="right",
        ) # score as text on the right bottom of the subgraph
        i += 1

    i = 0
    ax = axs[1][i]
    ax.axis('off') # turn off both x-axis and y-axis, including their labels and ticks
    i += 1

    # iterate over classifiers
    for name in models.keys():
        ax = axs[1][i]
        x = np.linspace(x_min, x_max, n_ticks)
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()] # same as above
        response = get_BI(X_grid, models[name])
        if rescale:
            vmin = min(response)
            vmax = max(response)
            response = (response-vmin) / (vmax-vmin)
        response = response.reshape(xs.shape)

        with np.errstate(all='ignore'):
            sns.heatmap(response, ax=ax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

    plt.subplots_adjust(#left=0.1,
                    bottom=.17,
                    #right=0.9,
                    top=.9,
                    wspace=0.,
                    hspace=0.)

    return figure

def diff_variances():

    figure = 0
    return figure