import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
from scipy.stats import beta
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.inspection._plot.decision_boundary import _check_boundary_response_method
from threadpoolctl import threadpool_limits

from ppu.methods.beta import get_Beta, get_Beta_para
from ppu.methods.utils import BS_models, get_dataset, get_models


def draw_beta_bs(classifier, generator, n_samples, bootstrap, n_models=64, threshold=0.5):
    cbar_kws = {"use_gridspec": False, "location": "bottom"}
    plt.rcParams.update({"font.size": 15})

    if not isinstance(bootstrap, list):
        bootstrap = [bootstrap]
    num = len(bootstrap)
    figure, axs = plt.subplots(2, num+1, figsize=(3*num+3, 7))

    i = 0
    eps = 1
    n_ticks = 100


    dataset = get_dataset(0, generator, n_samples=n_samples)
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
    # response1 = get_BI(X_grid, sep_model)
    response1 = get_Beta(X_grid, sep_model, threshold)
    vmin = min(response1)
    vmax = max(response1)
    response1 = response1.reshape(xs.shape)

    response2 = [None for j in range(num)]
    for i in range(num):
        with threadpool_limits(limits=4):
            #response2[i] = get_BI(X_grid, BS_models(func, gen, reps=bootstrap[i], n_samples=n_samples))
            response2[i] = get_Beta(X_grid, BS_models(classifier, generator, reps=bootstrap[i], n_samples=n_samples), threshold)
        vmin = min(vmin, min(response2[i]))  # noqa: PLW3301
        vmax = max(vmax, max(response2[i]))  # noqa: PLW3301
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


def draw_beta_pdf(start, end, classifier, generator, n_samples, n_models=64, threshold=0.5, n_pdf = 50):
    with threadpool_limits(limits=4):
        sep_model = get_models(classifier, generator, reps=n_models, n_samples=n_samples)
    paras = get_Beta_para(start, end, sep_model, threshold, num_points=n_pdf)

    x_line = np.linspace(start[0], end[0], len(paras))

    x_vals = []
    y_vals = []
    z_vals = []

    # 逐点绘制Beta分布的pdf
    for i, (a, b) in enumerate(zip(paras[:, 0], paras[:, 1])):
        x_pdf = np.linspace(0, 1, 100)
        y_pdf = beta.pdf(x_pdf, a, b)
        z = np.full_like(x_pdf, x_line[i % len(x_line)])  # 将线段上的点与Beta分布的pdf对应
        x_vals.append(z)
        y_vals.append(x_pdf)
        z_vals.append(y_pdf)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])

    surface = go.Surface(
        x = x_vals,
        y = y_vals,
        z = z_vals,
        colorscale="Viridis"
        )

    fig.add_trace(surface)

    fig.update_layout(
        scene={
            "xaxis_title": "Line Segment Points",
            "yaxis_title": "y",
            "zaxis_title": "PDF"
        }
    )

    return fig
