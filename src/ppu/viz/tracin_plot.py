import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from ppu.methods.tracin import get_random_resampled_tracin, get_random_tracin
from ppu.methods.utils import get_dataset, get_models


def draw(data, x, y, mode="2d", ax=None, contour=False, training_data=None):
    if mode == "2d":
        cbar_kws = {"use_gridspec": False, "location": "right"}
        if ax is None:
            ax = sns.heatmap(data, cbar_kws=cbar_kws)
        else:
            sns.heatmap(data, cbar_kws=cbar_kws, ax=ax)
        ax.set_xticks(np.arange(0, data.shape[1]+1, 10))
        ax.set_yticks(np.arange(0, data.shape[0]+1, 10))

        if isinstance(contour, (int, float)):
            X, Y = np.meshgrid(np.arange(data.shape[1])+0.5, np.arange(data.shape[0])+0.5)
            ax.contour(X, Y, data, levels=[contour], colors="white", linewidths=1.5)

        if training_data is not None:
            # Convert training data coordinates to heatmap coordinates
            scatter_x = np.interp(training_data[0][:, 0], x, np.arange(len(x)))
            scatter_y = np.interp(training_data[0][:, 1], y, np.arange(len(y)))
            ax.scatter(scatter_x, scatter_y,
                       c=training_data[1], cmap=ListedColormap(["#FF0000", "#0000FF"]),
                       edgecolors="k", alpha=0.4)


        ax.set_xticklabels([f"{x:.3f}" for x in np.linspace(x[0], x[-1], 11)])
        ax.set_yticklabels([f"{x:.3f}" for x in np.linspace(y[0], y[-1], 11)])

        ax.invert_yaxis()

        ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    elif mode == "3d":
        xs, ys = np.meshgrid(x, y)
        if ax is None:
            ax = plt.plot_surface(xs, ys, data, cmap="winter", edgecolor="none")
            ax.contourf(xs, ys, data, zdir='z', offset=np.min(data)-0.5, cmap="winter")
        else:
            ax.plot_surface(xs, ys, data, cmap="winter", edgecolor="none")
            ax.contourf(xs, ys, data, zdir='z', offset=np.min(data)-0.5, cmap="winter")
        ax.set_zlim(np.min(data)-0.5, np.max(data))
    return ax


def tracin_plot_iter(gen, n_samples, classifier, iters, ratio=0.25, **kwargs):
    figure, axs = plt.subplots(6, 2, figsize=(12,30))


    seed = 2

    data = get_dataset(seed, gen, n_samples=n_samples, **kwargs)
    nn  = get_models(classifier, gen, reps=1, n_samples=n_samples, seeds=[seed], **kwargs)[0]

    (X_train, y_train), (X_test, y_test) = data
    eps = 1.
    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

    n_ticks = 100
    x = np.linspace(
            x_min, x_max, n_ticks
        )  # for the heatmap we need to creat the grid first, so n_ticks are the density of the grid nodes
    y = np.linspace(y_min, y_max, n_ticks)
    xs, ys = np.meshgrid(x, y)
    X_grid = np.c_[xs.ravel(), ys.ravel()]


    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # color for data points
    ax = axs[0][0]  # position of subgraph
    ax.set_title("Data")  # subgraph title
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.set_xlim(x_min, x_max)  # axis range
    ax.set_ylim(y_min, y_max)

    ax = axs[0][1]
    ax.axis("off")

    tracin_0 = get_random_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(iters), batch_size=int(n_samples*ratio)).T
    tracin_1 = get_random_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(iters), batch_size=int(n_samples*ratio)).T

    tracin_0_resampled = get_random_resampled_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(iters), batch_size=int(n_samples*ratio)).T
    tracin_1_resampled = get_random_resampled_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(iters), batch_size=int(n_samples*ratio)).T


    for i in range(len(iters)):


        #loss0_1, loss0_2, tracin_0 = get_tracin(X_grid, 0., nn, train=(X_train, y_train), iter=iters[i], mode="random", num=500)
        #loss1_1, loss1_2, tracin_1 = get_tracin(X_grid, 1., nn, train=(X_train, y_train), iter=iters[i], mode="random", num=500)
        tracin_i = tracin_0[iters[i]-1] + tracin_1[iters[i]-1]

        tracin_i = tracin_i.reshape(xs.shape)

        ax = axs[i+1][0]
        draw(tracin_i, x, y, ax=ax)


        #loss0_1, loss0_2, tracin_0 = get_tracin(X_grid, 0., nn, train=(X_train, y_train), iter=iters[i], mode="random", num=500, resample=True)
        #loss1_1, loss1_2, tracin_1 = get_tracin(X_grid, 1., nn, train=(X_train, y_train), iter=iters[i], mode="random", num=500, resample=True)
        tracin_resampled_i = tracin_0_resampled[iters[i]-1] + tracin_1_resampled[iters[i]-1]

        tracin_i = tracin_i.reshape(xs.shape)

        ax = axs[i+1][1]
        draw(tracin_resampled_i, x, y, ax=ax)

    return figure


def tracin_plot_contour(gen, itrs, n_samples, classifier, resample=False, ratio=0.25, seed=0, eps=1., scatter=True, **kwargs):
    n_samples = 2000

    figure, axs = plt.subplots(len(itrs), 3, figsize=(15, 4.5*len(itrs)))

    data = get_dataset(seed, gen, n_samples=n_samples, **kwargs)
    nn  = get_models(classifier, gen, reps=1, n_samples=n_samples, seeds=[seed], **kwargs)[0]

    (X_train, y_train), (X_test, y_test) = data
    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

    n_ticks =100
    x = np.linspace(
            x_min, x_max, n_ticks
        )  # for the heatmap we need to creat the grid first, so n_ticks are the density of the grid nodes
    y = np.linspace(y_min, y_max, n_ticks)
    xs, ys = np.meshgrid(x, y)
    X_grid = np.c_[xs.ravel(), ys.ravel()]

    if resample:
        tracin_0 = get_random_resampled_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T
        tracin_1 = get_random_resampled_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T
    else:
        tracin_0 = get_random_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T
        tracin_1 = get_random_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T

    if resample:
        tracino_t = get_random_resampled_tracin(X_test=X_train, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T
        tracini_t = get_random_resampled_tracin(X_test=X_train, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T
    else:
        tracino_t = get_random_tracin(X_test=X_train, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T
        tracini_t = get_random_tracin(X_test=X_train, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=max(itrs), batch_size=int(n_samples*ratio)).T

    for i in range(len(itrs)):
        itr = itrs[i]



        #loss0_1, loss0_2, tracin_0 = get_tracin(X_grid, 0., nn, train=(X_train, y_train), iter=itr, mode="random", num=int(n_samples/4), resample=resample)
        #loss1_1, loss1_2, tracin_1 = get_tracin(X_grid, 1., nn, train=(X_train, y_train), iter=itr, mode="random", num=int(n_samples/4), resample=resample)
        tracin = tracin_0[itrs[i]-1] + tracin_1[itrs[i]-1]

        tracin = tracin.reshape(xs.shape)


        ax = axs[i][1]
        ax.hist(tracin, bins="auto", log=True)



        #losso_t_1, losso_t_2, tracino_t = get_tracin(X_train, 1., nn, train=(X_train, y_train), iter=itr, mode="random", num=int(n_samples/4), resample=resample)
        #lossi_t_1, lossi_t_2, tracini_t = get_tracin(X_train, 0., nn, train=(X_train, y_train), iter=itr, mode="random", num=int(n_samples/4), resample=resample)
        tracin_t = tracino_t[itrs[i]-1] + tracini_t[itrs[i]-1]

        ax = axs[i][2]
        ax.hist(tracin_t, bins="auto", log=True)

        indices = np.argsort(tracin_t)[-20:][::-1]
        head = tracin_t[indices]

        ax = axs[i][0]
        if scatter is True:
            draw(tracin, x, y, ax=ax, contour=head.mean(), training_data=[X_train, y_train])
        else:
            draw(tracin, x, y, ax=ax, contour=head.mean())
        # print(itr)

    plt.subplots_adjust(left=0.11,
            bottom=0.12,
            right=0.85,
            top=0.88,
            wspace=0.45,
            hspace=0.3,
        )

    return figure


def tracin_plot_single(gen, n_samples, classifier, itr, seed=0, resample=False, ratio=0.25, **kwargs):
    data = get_dataset(seed, gen, n_samples=n_samples, **kwargs)
    nn  = get_models(classifier, gen, reps=1, n_samples=n_samples, seeds=[seed], **kwargs)[0]

    (X_train, y_train), (X_test, y_test) = data
    eps = 1.
    x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
    y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

    n_ticks = 100
    x = np.linspace(
            x_min, x_max, n_ticks
        )  # for the heatmap we need to creat the grid first, so n_ticks are the density of the grid nodes
    y = np.linspace(y_min, y_max, n_ticks)
    xs, ys = np.meshgrid(x, y)
    X_grid = np.c_[xs.ravel(), ys.ravel()]

    if resample:
        tracin_0 = get_random_resampled_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracin_1 = get_random_resampled_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
    else:
        tracin_0 = get_random_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracin_1 = get_random_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T

    #loss0_1, loss0_2, tracin_0 = get_tracin(X_grid, 0., nn, train=(X_train, y_train), iter=itr, mode="random", num=500, resample=resample)
    #loss1_1, loss1_2, tracin_1 = get_tracin(X_grid, 1., nn, train=(X_train, y_train), iter=itr, mode="random", num=500, resample=resample)
    tracin = tracin_0[-1] + tracin_1[-1]

    if resample:
        tracino_t = get_random_resampled_tracin(X_test=X_train, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracini_t = get_random_resampled_tracin(X_test=X_train, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
    else:
        tracino_t = get_random_tracin(X_test=X_train, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracini_t = get_random_tracin(X_test=X_train, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T

    #losso_t_1, losso_t_2, tracino_t = get_tracin(X_train, 1., nn, train=(X_train, y_train), iter=itr, mode="random", num=500, resample=resample)
    #lossi_t_1, lossi_t_2, tracini_t = get_tracin(X_train, 0., nn, train=(X_train, y_train), iter=itr, mode="random", num=500, resample=resample)
    tracin_t = tracino_t[-1] + tracini_t[-1]

    tracin_0_i = tracin_0[-1].reshape(xs.shape)
    tracin_1_i = tracin_1[-1].reshape(xs.shape)
    tracin = tracin.reshape(xs.shape)

    figure, axs = plt.subplots(2, 3, figsize=(15, 7))
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # color for data points
    ax = axs[0][0]  # position of subgraph
    ax.set_title("Data")  # subgraph title
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.set_xlim(x_min, x_max)  # axis range
    ax.set_ylim(y_min, y_max)

    ax = axs[0][1]
    ax.hist(tracin, bins="auto", log=True)
    ax = axs[0][2]
    ax.hist(tracin_t, bins="auto", log=True)


    ax = axs[1][0]
    draw(tracin_0_i, x, y, ax=ax)
    ax = axs[1][1]
    draw(tracin_1_i, x, y, ax=ax)
    ax = axs[1][2]
    draw(tracin, x, y, ax=ax)



    plt.subplots_adjust(left=0.12,
            bottom=0.1,
            right=0.85,
            top=0.9,
            wspace=0.54,
            hspace=0.2,
        )

    return figure


def tracin_plot_seeds(gen, n_samples, classifier, seeds, itr=3, ratio=0.25, mode = "2d", **kwargs):
    if mode == "3d":
        figure, axs = plt.subplots(len(seeds), 3, figsize=(15, 4.5*len(seeds)), subplot_kw = {"projection": "3d"})
    elif mode == "2d":
        figure, axs = plt.subplots(len(seeds), 3, figsize=(15, 4.5*len(seeds)))


    for i in range(len(seeds)):
        seed = seeds[i]

        data = get_dataset(seed, gen, n_samples=n_samples, **kwargs)
        nn  = get_models(classifier, gen, reps=1, n_samples=n_samples, seeds=[seed], **kwargs)[0]

        (X_train, y_train), (X_test, y_test) = data
        eps = 1.
        x_min, x_max = X_train[:, 0].min() - eps, X_train[:, 0].max() + eps
        y_min, y_max = X_train[:, 1].min() - eps, X_train[:, 1].max() + eps

        n_ticks = 100
        x = np.linspace(
                x_min, x_max, n_ticks
            )  # for the heatmap we need to creat the grid first, so n_ticks are the density of the grid nodes
        y = np.linspace(y_min, y_max, n_ticks)
        xs, ys = np.meshgrid(x, y)
        X_grid = np.c_[xs.ravel(), ys.ravel()]


        cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # color for data points
        if mode == "3d":
            axs[i][0].remove()
            axs[i][0] = figure.add_subplot(4, 3, i*3+1)
        ax = axs[i][0]  # position of subgraph
        ax.set_title("Data")  # subgraph title
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.set_xlim(x_min, x_max)  # axis range
        ax.set_ylim(y_min, y_max)


        #loss0_1, loss0_2, tracin_0 = get_tracin(X_grid, 0., nn, train=(X_train, y_train), iter=itr, mode="random", num=500)
        #loss1_1, loss1_2, tracin_1 = get_tracin(X_grid, 1., nn, train=(X_train, y_train), iter=itr, mode="random", num=500)

        tracin_0 = get_random_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracin_1 = get_random_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracin = tracin_0[-1] + tracin_1[-1]

        tracin = tracin.reshape(xs.shape)

        ax = axs[i][1]
        draw(tracin, x, y, mode=mode, ax=ax)


        #loss0_1, loss0_2, tracin_0 = get_tracin(X_grid, 0., nn, train=(X_train, y_train), iter=itr, mode="random", num=500, resample=True)
        #loss1_1, loss1_2, tracin_1 = get_tracin(X_grid, 1., nn, train=(X_train, y_train), iter=itr, mode="random", num=500, resample=True)
        tracin_0_resampled = get_random_resampled_tracin(X_test=X_grid, y_test=0, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracin_1_resampled = get_random_resampled_tracin(X_test=X_grid, y_test=1, X_train=X_train, y_train=y_train, mlp=nn, n_iter=itr, batch_size=int(n_samples*ratio)).T
        tracin_resampled = tracin_0_resampled[-1] + tracin_1_resampled[-1]

        tracin_resampled = tracin_resampled.reshape(xs.shape)

        ax = axs[i][2]
        draw(tracin_resampled, x, y, mode=mode, ax=ax)

        # print(i)

    figure.suptitle(f"iter = {itr}", fontsize=30)

    plt.subplots_adjust(left=0.11,
            bottom=0.12,
            right=0.85,
            top=0.88,
            wspace=0.45,
            hspace=0.3,
        )

    return figure
