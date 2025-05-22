import heapq
import math

import numpy as np
from scipy.interpolate import interp1d


## Euclidian distance :
def euclidian_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def resample_path(path, n_points=500):
    distances = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative[-1]
    fx = interp1d(cumulative, path[:, 0], kind="linear")
    fy = interp1d(cumulative, path[:, 1], kind="linear")
    fz = interp1d(cumulative, path[:, 2], kind="linear")
    new_distances = np.linspace(0, total_length, n_points)
    return np.stack((fx(new_distances), fy(new_distances),fz(new_distances)), axis=1), np.sum(distances)


def line_of_sight(p1, p2, Z, min_distance):
    x_vals = np.linspace(p1[0], p2[0], 100)
    y_vals = np.linspace(p1[1], p2[1], 100)

    x_idx = np.clip(np.round(x_vals).astype(int), 0, Z.shape[1] - 1)
    y_idx = np.clip(np.round(y_vals).astype(int), 0, Z.shape[0] - 1)
    values = Z[x_idx, y_idx]
    return np.all(values >= min_distance)


def shortcut_path(path_indices, X, Y, Z, min_distance):
    shortcut = [path_indices[0]]
    i = 0
    while i < len(path_indices) - 1:
        j = len(path_indices) - 1
        while j > i + 1:
            if line_of_sight(path_indices[i], path_indices[j], Z, min_distance):
                break
            j -= 1
        path_indices[j]
        shortcut.append(path_indices[j])
        i = j
    return np.array(shortcut)
