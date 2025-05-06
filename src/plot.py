import matplotlib.pyplot as plt
import numpy as np

colors_default = plt.cm.tab10.colors
from .generate_path import (
    generate_curve,
    generate_demi_circle_path,
    generate_random_ondulating_path,
)
from .simulation import rankine_vortex, uniform_velocity


def plot_trajectories(
    ax,
    trajectories_list,
    path,
    title,
    a=0,
    center=np.zeros(2),
    cir=0,
    dir=np.zeros(2),
    norm=0,
    plot_background=False,
    type="",
    color_id=0,
    colors=plt.cm.tab10.colors,
    label="",
):
    if colors is None:
        colors = colors_default
    if isinstance(trajectories_list[0][0], np.ndarray):
        for idx, list_state in enumerate(trajectories_list):
            """indices = np.linspace(0, len(path) - 1, list_state[1]).astype(int)
            path_sampled = path[indices]
            ax.plot(path_sampled[:, 0], path_sampled[:, 1], label='path', color='black', linewidth=2)
            """
            states = list_state[0]

            color_id_t = max(idx, color_id)
            ax.plot(
                states[:, 0],
                states[:, 1],
                color=colors[color_id_t],
                linewidth=0.9,
                label=label,
            )
            ax.scatter(states[-1, 0], states[-1, 1], color=colors[color_id_t], s=5)
            ax.scatter(states[0, 0], states[0, 1], color=colors[color_id_t], s=5)
            ax.set_aspect("equal")
    else:
        states = trajectories_list
        color_id_t = color_id
        ax.plot(
            states[:, 0],
            states[:, 1],
            color=colors[color_id_t],
            linewidth=0.9,
            label=label,
        )
        ax.scatter(states[-1, 0], states[-1, 1], color=colors[color_id_t], s=5)
        ax.scatter(states[0, 0], states[0, 1], color=colors[color_id_t], s=5)
        ax.set_aspect("equal")
    if plot_background:
        x_bound = ax.get_xlim()
        y_bound = ax.get_ylim()
        plot_background_velocity(type, x_bound, y_bound, a, center, cir, dir, norm)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title != "":
        ax.set_title(f"{title}")
    else:
        if type == "uniform":
            ax.set_title(f"Trajectories - norm : {norm}")
        if type == "rankine":
            ax.set_title(f"Trajectories - a : {a} - circulation : {cir}")


def plot_action(path, x, p_0, id_cp, action, id):
    plt.scatter(x[0], x[1], color=colors_default[id % 10])
    plt.annotate(
        f"{id}", xy=(x[0], x[1]), xytext=(x[0], x[1] + 1 / (64 * 20))  # texte en LaTeX
    )
    plt.scatter(
        path[id_cp, 0], path[id_cp, 1], color=colors_default[id % 10], marker="*"
    )
    plt.quiver(x[0], x[1], action[0], action[1], scale=20, width=0.005, color="grey")
    plt.xlabel("x")
    plt.ylabel("y")


def plot_background_velocity(
    type,
    x_bound,
    y_bound,
    a=0.25,
    center=(0.5, 0.5),
    cir=0.8,
    dir=np.zeros(2),
    norm=0.0,
):
    x = np.linspace(x_bound[0], x_bound[1], 10)
    y = np.linspace(y_bound[0], y_bound[1], 10)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if type == "rankine":
                v = rankine_vortex((X[i, j], Y[i, j]), a, center, cir)
                U[i, j] = v[0]
                V[i, j] = v[1]
            if type == "uniform":
                v = uniform_velocity(dir, norm)
                U[i, j] = v[0]
                V[i, j] = v[1]
    plt.quiver(X, Y, U, V, scale=15, width=0.002, color="gray")
    if type == "rankine":
        plt.scatter(center[0], center[1], marker="*")
    plt.xlabel("x")
    plt.ylabel("y")
