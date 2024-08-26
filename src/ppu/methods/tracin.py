import contextlib
import copy
import random

import numpy as np
import torch
from sklearn.neighbors import KDTree


def get_tracin(X, label, nn, iter=1, train=None, mode="random", num=None, resample=False):
    loss1 = []
    loss2 = []
    tracin = []
    if num is None:
        if mode == "neighbor":
            num = 10
        elif mode in ("random", "uniform"):
            num = int(len(X) / 4)
    if not isinstance(label, np.ndarray):
        label = np.array([label for i in range(len(X))])
    if train is None:
        for i in range(len(X)):
            x = torch.tensor(X[i]).to(torch.float32)
            label_t = torch.tensor(label[i]).to(torch.float32)
            output1 = nn.model(x)
            loss1.append(nn.criterion(output1.squeeze(-1), label_t).detach().cpu().numpy())
            nn_next = copy.deepcopy(nn)
            for j in range(iter):
                nn_next.train_epoch(x, label_t)
            output2 = nn_next.model(x)
            loss2.append(nn_next.criterion(output2.squeeze(-1), label_t).detach().cpu().numpy())
            del nn_next
            tracin.append(loss1[-1] - loss2[-1])
    else:
        for i in range(len(X)):
            output1 = nn.model(torch.tensor(X[i]).to(torch.float32).to(device=nn.device))
            loss1.append(nn.criterion(output1.squeeze(-1), torch.tensor(label[i]).to(torch.float32).to(device=nn.device)).detach().cpu().numpy())
            nn_next = copy.deepcopy(nn)
            if mode == "neighbor":
                tree = KDTree(train[0])
                dist, ind = tree.query(X[[i]], k=num)
                x = np.append(train[0][ind[0]], X[[i]], axis=0)
                x = torch.from_numpy(x).to(dtype=torch.float32)
                label_t = np.append(train[1][ind[0]], label[i])
                label_t = torch.from_numpy(label_t).to(dtype=torch.float32)
            elif mode == "random":
                random.seed(0)
                ind = random.sample(range(len(train[0])), num)
                x = np.append(train[0][ind], X[[i]], axis=0)
                x = torch.from_numpy(x).to(dtype=torch.float32)
                label_t = np.append(train[1][ind], label[i])
                label_t = torch.from_numpy(label_t).to(dtype=torch.float32)
            elif mode == "uniform":
                x_min, x_max = train[0][:, 0].min(), train[0][:, 0].max()
                y_min, y_max = train[0][:, 1].min(), train[0][:, 1].max()
                grid_size = (x_max - x_min) / int(num**0.5)
                grid = {}
                # Assign points to grid cells
                for idx, point in enumerate(train[0]):
                    x_idx = int((point[0] - x_min) / grid_size)
                    y_idx = int((point[1] - y_min) / grid_size)
                    grid_cell = (x_idx, y_idx)
                    if grid_cell not in grid:
                        grid[grid_cell] = []
                    grid[grid_cell].append((point, idx))
                # Select one point per occupied grid cell and get their indices
                ind = [pts[0][1] for pts in grid.values()]
                x = torch.from_numpy(train[0][ind]).to(dtype=torch.float32)
                label_t = torch.from_numpy(train[1][ind]).to(dtype=torch.float32)
            for j in range(iter):
                if resample and mode != "neighbor":
                    if mode == "random":
                        random.seed(j + 1)
                        ind = random.sample(range(len(train[0])), num)
                        x = np.append(train[0][ind], X[[i]], axis=0)
                        x = torch.from_numpy(x).to(dtype=torch.float32)
                        label_t = np.append(train[1][ind], label[i])
                        label_t = torch.from_numpy(label_t).to(dtype=torch.float32)
                    elif mode == "uniform":
                        x_min, x_max = train[0][:, 0].min(), train[0][:, 0].max()
                        y_min, y_max = train[0][:, 1].min(), train[0][:, 1].max()
                        grid_size = (x_max - x_min) / int(num**0.5)
                        grid = {}
                        # Assign points to grid cells
                        for idx, point in enumerate(train[0]):
                            x_idx = int((point[0] - x_min) / grid_size)
                            y_idx = int((point[1] - y_min) / grid_size)
                            grid_cell = (x_idx, y_idx)
                            if grid_cell not in grid:
                                grid[grid_cell] = []
                            grid[grid_cell].append((point, idx))
                        # Select one point per occupied grid cell and get their indices
                        ind = []
                        for pts in grid.values():
                            for k in range(j):
                                with contextlib.suppress(Exception):
                                    idx = pts[j-k][1]
                            ind.append(idx)
                        x = torch.from_numpy(train[0][ind]).to(dtype=torch.float32)
                        label_t = torch.from_numpy(train[1][ind]).to(dtype=torch.float32)
                nn_next.train_epoch(x, label_t)
            output2 = nn_next.model(torch.tensor(X[i]).to(torch.float32).to(device=nn.device))
            loss2.append(nn_next.criterion(output2.squeeze(-1), torch.tensor(label[i]).to(torch.float32).to(device=nn.device)).detach().cpu().numpy())
            del nn_next
            tracin.append(loss1[-1] - loss2[-1])
    return np.array(loss1), np.array(loss2), np.array(tracin)
