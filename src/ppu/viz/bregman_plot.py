import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from threadpoolctl import threadpool_limits

from ppu.methods.bregman import get_BI
from ppu.methods.utils import BS_models, get_dataset, get_models


def draw_classifier(generator, n_samples, names, classifiers, models, rescale = False):
    dataset = get_dataset(1000, generator, n_samples=n_samples)
    offset = 1
    rows = 2

    gridspec_kw={"height_ratios": [1, 1]}
    cbar_kws = {"use_gridspec": False, "location": "bottom"}
    plt.rcParams.update({"font.size": 15})
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

        response = _check_boundary_response_method(clf, "auto")(X_grid) # the function returns the probability of points(inputs) belong to each class

        if len(response.shape) != 1:
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
    ax.axis("off") # turn off both x-axis and y-axis, including their labels and ticks
    i += 1

    # iterate over classifiers
    for name in models:
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

        with np.errstate(all="ignore"):
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


def draw_classifiers_BIbs(generator, n_samples, selected_clfs, selected_models, n_models=64, rescale=False):
    dataset = get_dataset(1000, generator, n_samples=n_samples)
    cbar_kws = {"use_gridspec": False, "location": "bottom"}
    plt.rcParams.update({"font.size": 15})
    figure, axs = plt.subplots(2, 5, figsize=(15, 7))


    i = 0
    eps = 1
    n_ticks = 100

    ds = dataset
    (X_train, y_train), (X_test, y_test) = ds # X is the data point, y is the class label

    # the area we draw is a bit larger than the range of data points
    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

    # just plot the dataset first
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

    for name in selected_models:
        ax = axs[0][i]
        ax.set_title(name)
        x = np.linspace(x_min, x_max, n_ticks)
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()] # same as above
        response = get_BI(X_grid, selected_models[name])
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


    i = 0
    ax = axs[1][i]
    ax.axis('off')
    i += 1

    for name in selected_models:
        x = np.linspace(x_min, x_max, n_ticks)
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()]

        with threadpool_limits(limits=1):
            response = get_BI(X_grid, BS_models(selected_clfs[name], generator, reps=n_models, n_samples=n_samples))
        response = response.reshape(xs.shape)
        ax = axs[1][i]

        with np.errstate(all='ignore'):
            sns.heatmap(response, ax=ax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

    plt.subplots_adjust(#left=0.1,
                        bottom=.2,
                        #right=0.9,
                        top=.9,
                        wspace=0.1,
                        hspace=0.3)

    return figure


def draw_diff_variances(classifier, generator, n_samples, variances, n_models=64):
    cbar_kws = {"use_gridspec": False, "location": "bottom"}
    plt.rcParams.update({"font.size": 15})
    n_fig = len(variances)
    figure, axs = plt.subplots(3, n_fig, figsize=(n_fig * 3, 10.5))
    i = 0
    eps = 1
    n_ticks = 100

    for i in range(n_fig):
        dataset = get_dataset(1000, generator, n_samples=n_samples, scale=(variances[i], variances[i]), class_sep=1.7)
        ds = dataset

        (X_train, y_train), (X_test, y_test) = ds # X is the data point, y is the class label

        with threadpool_limits(limits=1):
            sep_model = get_models(classifier, generator, reps=n_models, n_samples=n_samples, scale=(variances[i], variances[i]), class_sep=1.7)

        # the area we draw is a bit larger than the range of data points
        x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
        y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

        cm_bright = ListedColormap(["#FF0000", "#0000FF"]) # color for data points 
        ax = axs[0][i] # position of subgraph
        ax.set_title(f"Var={variances[i]}") # subgraph title
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.set_xlim(x_min, x_max) # axis range
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        x = np.linspace(x_min, x_max, n_ticks)
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()]
        response1 = get_BI(X_grid, sep_model)
        vmin = min(response1)
        vmax = max(response1)
        response1 = response1.reshape(xs.shape)

        with threadpool_limits(limits=1):
            bs_models = BS_models(classifier, generator, reps=n_models, n_samples=n_samples, scale=(variances[i], variances[i]), class_sep=1.7)

        response2 = get_BI(X_grid, bs_models)
        vmin = min(vmin, *response2)
        vmax = max(vmax, *response2)
        response2 = response2.reshape(xs.shape)

        ax = axs[1][i]
        #ax.set_title()
        with np.errstate(all='ignore'):
            sns.heatmap(response1, ax=ax, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())


        ax = axs[2][i]

        with np.errstate(all='ignore'):
            sns.heatmap(response2, ax=ax, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())

        i += 1

    plt.subplots_adjust(#left=0.1,
                bottom=.15,
                #right=0.9,
                top=.9,
                wspace=0.1,
                hspace=0.3)

    return figure


def draw_diff_class_sep(classifier, generator, n_samples, seps, n_models=64):
    cbar_kws = dict(use_gridspec=False, location="bottom")
    plt.rcParams.update({'font.size': 15})
    n_fig = len(seps)
    figure, axs = plt.subplots(3, n_fig, figsize=(3 * n_fig, 10.5))

    i = 0
    eps = 1
    n_ticks = 100


    for i in range(n_fig):
        dataset = get_dataset(1000, generator, n_samples=n_samples, class_sep=seps[i])
        ds = dataset

        (X_train, y_train), (X_test, y_test) = ds # X is the data point, y is the class label

        with threadpool_limits(limits=4):
            sep_model = get_models(classifier, generator, reps=n_models, n_samples=n_samples, class_sep=seps[i])

        # the area we draw is a bit larger than the range of data points
        x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
        y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

        cm_bright = ListedColormap(["#FF0000", "#0000FF"]) # color for data points
        ax = axs[0][i] # position of subgraph
        ax.set_title(f"sep={seps[i]}") # subgraph title
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.set_xlim(x_min, x_max) # axis range
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        x = np.linspace(x_min, x_max, n_ticks)
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()]
        response1 = get_BI(X_grid, sep_model)
        vmin = min(response1)
        vmax = max(response1)
        response1 = response1.reshape(xs.shape)


        with threadpool_limits(limits=4):
            response2 = get_BI(X_grid, BS_models(classifier, generator, reps=n_models, n_samples=n_samples, class_sep=seps[i]))
        vmin = min(vmin, min(response2))
        vmax = max(vmax, max(response2))
        response2 = response2.reshape(xs.shape)

        ax = axs[1][i]
        #ax.set_title()
        with np.errstate(all='ignore'):
            sns.heatmap(response1, ax=ax, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())


        ax = axs[2][i]

        with np.errstate(all='ignore'):
            sns.heatmap(response2, ax=ax, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())

        i += 1


    plt.subplots_adjust(#left=0.1,
                        bottom=.2,
                        #right=0.9,
                        top=.9,
                        wspace=0.1,
                        hspace=0.5)

    return figure


def draw_diff_bootstrap(classifier, generator, n_samples, bootstrap, n_models=64):
    cbar_kws = {"use_gridspec": False, "location": "bottom"}
    plt.rcParams.update({"font.size": 15})

    num = len(bootstrap)
    figure, axs = plt.subplots(2, num+1, figsize=(3*num+3, 7))

    i = 0
    eps = 1
    n_ticks = 100


    dataset = get_dataset(1000, generator, n_samples=n_samples)
    ds = dataset
    (X_train, y_train), (X_test, y_test) = ds # X is the data point, y is the class label

    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

    cm_bright = ListedColormap(["#FF0000", "#0000FF"]) # color for data points
    ax = axs[0][0] # position of subgraph
    ax.set_title("Data") # subgraph title
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.set_xlim(x_min, x_max) # axis range
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    ax = axs[1][0]
    ax.axis("off")

    with threadpool_limits(limits=4):
        sep_model = get_models(classifier, generator, reps=n_models, n_samples=n_samples)

    # the area we draw is a bit larger than the range of data points
    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps



    x = np.linspace(x_min, x_max, n_ticks)
    y = np.linspace(y_min, y_max, n_ticks)
    xs, ys = np.meshgrid(x, y)
    X_grid = np.c_[xs.ravel(), ys.ravel()]
    response1 = get_BI(X_grid, sep_model)
    vmin = min(response1)
    vmax = max(response1)
    response1 = response1.reshape(xs.shape)

    response2 = [None for j in range(num)]
    for i in range(num):
        with threadpool_limits(limits=4):
            response2[i] = get_BI(X_grid, BS_models(classifier, generator, reps=bootstrap[i], n_samples=n_samples))
        vmin = min(vmin, *response2[i])
        vmax = max(vmax, *response2[i])
        response2[i] = response2[i].reshape(xs.shape)

    for i in range(num):
        ax = axs[0][i+1]
        ax.set_title(f"bootstrap={bootstrap[i]}")
        with np.errstate(all="ignore"):
            sns.heatmap(response1, ax=ax, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())


        ax = axs[1][i+1]

        with np.errstate(all="ignore"):
            sns.heatmap(response2[i], ax=ax, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws).invert_yaxis()

        ax.set_xticks(())
        ax.set_yticks(())



    plt.subplots_adjust(#left=0.1,
                        bottom=.2,
                        #right=0.9,
                        top=.9,
                        wspace=0.1,
                        hspace=0.3)

    return figure
