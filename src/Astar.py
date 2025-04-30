import heapq
import math

import numpy as np
from scipy.interpolate import interp1d


## Euclidian distance :
def euclidian_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def heuristic(current, goal, grid):
    dist = euclidian_distance(current, goal)
    sdf_value = grid[current[0], current[1]]
    return dist * (1 + 0.05 / max(sdf_value, 0.001))


def visibility_heuristic(current, goal, grid, min_distance):
    euclidean_dist = euclidian_distance(current, goal)
    penalty_factor = 2
    if line_of_sight(current, goal, grid, min_distance):
        return euclidean_dist * penalty_factor
    return euclidean_dist


def astar(start, goal, grid, min_distance):
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, grid)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_neighbors(current):
            if is_valid_move(neighbor, grid, min_distance):
                tentative_g_score = g_score[current] + euclidian_distance(
                    current, neighbor
                )
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(
                        neighbor, goal, grid
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def get_neighbors(point):
    x, y = point
    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
        (x + 1, y + 1),
        (x - 1, y - 1),
        (x - 1, y + 1),
        (x + 1, y - 1),
    ]


def is_valid_move(point, grid, min_distance):
    x, y = point
    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
        return False
    if grid[x, y] < min_distance:
        return False
    return True


def plot_valid_invalid_points(grid, min_distance):
    import matplotlib.pyplot as plt

    valid_points = []
    invalid_points = []

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if is_valid_move((x, y), grid, min_distance):
                valid_points.append((x, y))
            else:
                invalid_points.append((x, y))

    valid_points = np.array(valid_points)
    invalid_points = np.array(invalid_points)

    plt.figure(figsize=(8, 8))
    if len(valid_points) > 0:
        plt.scatter(
            valid_points[:, 1],
            valid_points[:, 0],
            c="green",
            label="Valid Points",
            s=10,
        )
    if len(invalid_points) > 0:
        plt.scatter(
            invalid_points[:, 1],
            invalid_points[:, 0],
            c="red",
            label="Invalid Points",
            s=10,
        )
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Valid and Invalid Points")
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.show()


def resample_path(path, n_points=500):
    distances = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative[-1]
    fx = interp1d(cumulative, path[:, 0], kind="linear")
    fy = interp1d(cumulative, path[:, 1], kind="linear")
    new_distances = np.linspace(0, total_length, n_points)
    return np.stack((fx(new_distances), fy(new_distances)), axis=1)


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
