import math

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from src.Astar import astar, plot_valid_invalid_points, resample_path, shortcut_path
from matplotlib.contour import QuadContourSet

def sdf_circle(point, center, radius):
    px, py = point
    cx, cy = center
    distance_to_center = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return distance_to_center - radius


def sdf_many_circle(point, centers, radius):
    distances = np.zeros(len(centers))
    for i, center in enumerate(centers):
        distances[i] = np.linalg.norm(np.array(center) - np.array(point))

    id = np.argmin(distances)
    center = centers[id]
    return sdf_circle(point, center, radius)


def get_contour_coordinates(X, Y, Z, level=0):
    if not np.isfinite(Z).all():
        raise ValueError("Z contient des NaN ou des valeurs infinies.")
    if not (np.min(Z) <= level <= np.max(Z)):
        raise ValueError(
            f"Le niveau demandé ({level}) est en dehors des valeurs de Z (min: {np.min(Z)}, max: {np.max(Z)})"
        )

    contours = measure.find_contours(Z, level=level)
    if not contours:
        raise RuntimeError("Aucun contour trouvé pour ce niveau.")

    all_coords = []
    for contour in contours:
        i = contour[:, 0]  # ligne (axe y)
        j = contour[:, 1]  # colonne (axe x)
        x_coords = X[0, :]  # axe x
        y_coords = Y[:, 0]  # axe y
        x = np.interp(j, np.arange(X.shape[1]), x_coords)
        y = np.interp(i, np.arange(Y.shape[0]), y_coords)
        coords = np.stack((x, y), axis=-1)  # shape (M, 2)
        all_coords.append(coords)

    # Concaténer tous les points en un seul tableau de shape (N, 2)
    return np.concatenate(all_coords, axis=0)


def plot_sdf_path(X, Y, Z, path):
    plt.figure(figsize=(6, 6))

    plot_sdf(X, Y, Z)
    if path is not None:
        plt.plot(X[path[:, 0], path[:, 1]], Y[path[:, 0], path[:, 1]], color="black")

    else:
        print("No path found!")

    plt.title("SDF of a Circle with A* Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    # plt.legend()
    plt.savefig("fig/sdf_circle_path_plot.png", dpi=100, bbox_inches="tight")


def plot_sdf(X, Y, Z):
    contour = plt.contourf(X, Y, Z, levels=100, cmap="coolwarm")
    plt.colorbar(contour, label="Signed Distance")
    plt.contour(X, Y, Z, levels=[0], colors="black", linewidths=1)


if __name__ == "__main__":
    type = "circle"
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y, indexing="ij")
    print(X[0, 250])
    print(Y[0, 250])
    if type == "circle":
        centers = [(-1, 1 / 2), (1, -1 / 2)]
        radius = 3 / 4
        min_distance = 0.1
        Z = np.vectorize(lambda px, py: sdf_many_circle((px, py), centers, radius))(
            X, Y
        )
    start = [0, 500]
    goal = [999, 500]
    start_coords = (X[start[0], start[1]], Y[start[0], start[1]])
    goal_coords = (X[goal[0], goal[1]], Y[goal[0], goal[1]])
    print("Start coordinates:", start_coords)
    print("Goal coordinates:", goal_coords)
    path_indices = np.array(astar(start, goal, Z, min_distance))

    plot_sdf_path(X, Y, Z, path_indices)
