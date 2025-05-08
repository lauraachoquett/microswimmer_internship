import copy  # Assurez-vous que copy est importé
import os
import pickle  # Assurez-vous que pickle est importé
from math import atan2

import matplotlib.pyplot as plt
import numpy as np

from src.generate_path import generate_simple_line
from src.plot import plot_trajectories


def f(v, v_target, beta):
    return np.linalg.norm(v - v_target) + beta * abs(v[1])


def find_next_v(v_t, v_target, beta, Dt, num_angles=360):
    dir = v_target - v_t
    theta = atan2(dir[1], dir[0])
    alpha = np.pi / 8
    best_v = None
    best_f = np.inf
    best_dir = np.zeros(2)
    for angle in np.linspace(theta - alpha, theta + alpha, num_angles):
        candidate = v_t + Dt * np.array([np.cos(angle), np.sin(angle)])
        val = f(candidate, v_target, beta)
        if val < best_f:
            best_f = val
            best_v = candidate
            best_dir = np.array([np.cos(angle), np.sin(angle)])
    return best_v, best_dir


def visualize_streamline_analytic(p_0, p_target, beta, Dt, T, save_path_eval, title=""):
    save_path_streamline = os.path.join(save_path_eval, "streamlines/")
    if not os.path.exists(save_path_streamline):
        os.makedirs(save_path_streamline)
    p_target = p_target
    p_0 = p_0
    nb_points_path = 500
    nb_starting_point = 20
    p_0_above = p_0 + np.array([0, 0.2])
    p_target_above = p_target + np.array([0, 0.2])
    p_0_below = p_0 + np.array([0, -0.2])
    p_target_below = p_target + np.array([0, -0.2])

    path, _ = generate_simple_line(p_0, p_target, nb_points_path)
    path_above_point, _ = generate_simple_line(
        p_0_above, p_target_above, nb_starting_point
    )
    path_below_point, _ = generate_simple_line(
        p_0_below, p_target_below, nb_starting_point
    )

    path_above_point = path_above_point[:-1]
    path_below_point = path_below_point[:-1]
    path_starting_point = np.concatenate((path_above_point, path_below_point), axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(path[:, 0], path[:, 1], color="black", linewidth=2)

    for starting_p in path_starting_point:
        vs = [starting_p]
        for t in range(T):
            while np.linalg.norm(vs[-1] - p_target) > 0.07:
                v_t = vs[-1]
                v_next, _ = find_next_v(v_t, p_target, beta, Dt)
                vs.append(v_next)

        vs = np.array(vs)
        plot_trajectories(ax, vs, path, "streamlines")

    ax.axis("equal")
    ax.set_title(title)

    # Sauvegarde de la figure
    path_save_fig = os.path.join(save_path_streamline, f"analytic_streamline.png")
    fig.savefig(path_save_fig, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure sauvegardée à : {path_save_fig}")


if __name__ == "__main__":
    beta = 0.25
    Dt = 0.001
    T = 50

    p_0 = np.array([0, 0])
    p_target = np.array([2, 0])

    save_path_eval = "./fig"

    visualize_streamline_analytic(
        p_0, p_target, beta, Dt, T, save_path_eval, title="streamline"
    )
