import heapq
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from .data_loader import load_sdf_from_csv, vel_read
from .sdf import sdf_circle


def ordered_upwind_method_fast(x, y, v0, vx, vy, goal_point, narrow_band_width=5):
    """
    Version optimisée de l'Ordered Upwind Method pour grandes grilles.

    Paramètres additionnels:
    - narrow_band_width: Largeur de la bande étroite pour le Fast Marching Method

    Retourne:
    - travel_time: Champ des temps de trajet
    """
    nx, ny = len(x), len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialisation
    travel_time = np.full((ny, nx), np.inf)
    status = np.full((ny, nx), 0, dtype=np.int8)  # 0: far, 1: narrow, 2: accepted

    # Trouver l'indice du point d'arrivée
    i_goal = np.argmin(np.abs(x - goal_point[0]))
    j_goal = np.argmin(np.abs(y - goal_point[1]))

    travel_time[j_goal, i_goal] = 0.0
    status[j_goal, i_goal] = 2  # accepted

    # File de priorité pour les points 'narrow'
    heap = []

    # Directions (4-connexité)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Pré-calcul des vitesses effectives
    effective_speed = compute_effective_speeds(nx, ny, v0, vx, vy, dx, dy)
    print("Effective Speed")
    # Initialiser les voisins du point d'arrivée
    for di, dj in directions:
        ni, nj = i_goal + di, j_goal + dj
        if 0 <= ni < nx and 0 <= nj < ny:
            # Vérifier si la vitesse effective est positive
            speed_idx = get_direction_index(di, dj)
            if effective_speed[nj, ni, speed_idx] > 0:
                dist = np.sqrt((di * dx) ** 2 + (dj * dy) ** 2)
                t = dist / effective_speed[nj, ni, speed_idx]
                travel_time[nj, ni] = t
                status[nj, ni] = 1  # narrow
                heapq.heappush(heap, (t, ni, nj))

    # Boucle principale - Fast Marching
    while heap:
        time, i, j = heapq.heappop(heap)

        # Si le point est déjà accepté, continuer
        if status[j, i] == 2:  # accepted
            continue

        # Marquer comme accepté
        status[j, i] = 2  # accepted

        # Mettre à jour les voisins
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and status[nj, ni] != 2:  # not accepted
                speed_idx = get_direction_index(di, dj)
                if effective_speed[nj, ni, speed_idx] > 0:
                    # Calculer le nouveau temps depuis ce point
                    dist = np.sqrt((di * dx) ** 2 + (dj * dy) ** 2)
                    new_time = (
                        travel_time[j, i] + dist / effective_speed[nj, ni, speed_idx]
                    )

                    # Calculer aussi depuis les voisins acceptés du point (i,j)
                    for di2, dj2 in directions:
                        i2, j2 = i + di2, j + dj2
                        if (
                            (i2 != ni or j2 != nj)
                            and 0 <= i2 < nx
                            and 0 <= j2 < ny
                            and status[j2, i2] == 2
                        ):
                            # Interpoler le temps entre (i,j) et (i2,j2)
                            alpha = 0.5  # Simplification - utiliser un point médian
                            interp_i = i + di2 * alpha
                            interp_j = j + dj2 * alpha

                            # Distance du point interpolé au point cible
                            di_target = ni - interp_i
                            dj_target = nj - interp_j
                            dist_target = np.sqrt(
                                (di_target * dx) ** 2 + (dj_target * dy) ** 2
                            )

                            # Direction vers la cible
                            dir_idx = get_direction_index(
                                int(np.sign(di_target)), int(np.sign(dj_target))
                            )

                            # Vitesse effective le long de cette direction
                            interp_speed = effective_speed[nj, ni, dir_idx]
                            if interp_speed > 0:
                                # Temps depuis le début jusqu'au point interpolé
                                interp_time = (
                                    travel_time[j, i] * (1 - alpha)
                                    + travel_time[j2, i2] * alpha
                                )

                                # Temps total
                                t = interp_time + dist_target / interp_speed
                                new_time = min(new_time, t)

                    if new_time < travel_time[nj, ni]:
                        travel_time[nj, ni] = new_time
                        if status[nj, ni] == 0:  # far
                            status[nj, ni] = 1  # narrow
                            heapq.heappush(heap, (new_time, ni, nj))
                        else:  # status == 1 (narrow)
                            # Réinsérer avec le temps mis à jour
                            heapq.heappush(heap, (new_time, ni, nj))

    return travel_time


@njit
def get_direction_index(di, dj):
    """Convertit la direction (di, dj) en index pour le tableau des vitesses."""
    if di == -1 and dj == 0:
        return 0  # gauche
    elif di == 1 and dj == 0:
        return 1  # droite
    elif di == 0 and dj == -1:
        return 2  # bas
    elif di == 0 and dj == 1:
        return 3  # haut
    else:
        # Direction diagonale - approximer avec la direction principale
        if abs(di) > abs(dj):
            return 0 if di < 0 else 1
        else:
            return 2 if dj < 0 else 3


@njit(parallel=True)
def compute_effective_speeds(nx, ny, v0, vx, vy, dx, dy):
    """Pré-calcule les vitesses effectives dans les 4 directions principales."""
    # [gauche, droite, bas, haut]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    effective_speed = np.zeros((ny, nx, 4))

    for j in prange(ny):
        for i in range(nx):
            for k, (di, dj) in enumerate(directions):
                # Calculer la vitesse effective dans cette direction
                dir_norm = np.sqrt((di * dx) ** 2 + (dj * dy) ** 2)
                if dir_norm > 0:
                    dir_x = di * dx / dir_norm
                    dir_y = dj * dy / dir_norm
                    flow_component = vx[j, i] * dir_x + vy[j, i] * dir_y
                    effective_speed[j, i, k] = v0[j, i] / 4 + flow_component

    return effective_speed


def compute_fmm_path(
    start_point,
    goal_point,
    sdf_function,
    x,
    y,
    B,
    flow_field=None,
    grid_size=(200, 200),
    domain_size=(1.0, 1.0),
    ratio=1,
    flow_factor=1,
):
    """
    Calcule un chemin optimal en utilisant Fast Marching Method

    Arguments:
        start_point: Tuple (x, y) du point de départ
        goal_point: Tuple (x, y) du point d'arrivée
        sdf_function: Fonction qui retourne la SDF en chaque point (x, y)
        flow_field: Fonction qui retourne le vecteur d'écoulement (vx, vy) en chaque point, None si pas d'écoulement
        grid_size: Dimensions de la grille discrétisée
        domain_size: Dimensions physiques du domaine

    Returns:
        path: Liste de points [(x, y)] représentant le chemin optimal
    """
    if sdf_function(start_point) > 0 or sdf_function(goal_point) > 0:
        print(
            "Invalid starting point or target point :",
            sdf_function(start_point),
            sdf_function(goal_point),
        )

    X, Y = np.meshgrid(x, y)
    save_path_phi = f"data/phi/grid_size_{grid_size[0]}_{grid_size[1]}_phi.npy"
    if os.path.exists(save_path_phi):
        phi = np.load(save_path_phi)
        print("Phi loaded")
    else:
        phi = np.zeros((len(y), len(x)))
        for i in range(len(x)):
            for j in range(len(y)):
                a = sdf_function((x[i], y[j]))
                phi[j, i] = a
        phi = 3 * phi / np.max(np.abs(phi))
        os.makedirs(os.path.dirname(save_path_phi), exist_ok=True)
        np.save(save_path_phi, phi)
    print("SDF computed")
    speed = 1.0 / (1.0 + np.exp(B * phi))
    speed = np.clip(speed, 0.001, 1.0)
    save_path_flow = f"data/velocity_flow/grid_size_{grid_size[0]}_{grid_size[1]}/"
    if flow_field is not None:
        if os.path.exists(save_path_flow):
            flow_strength = ratio * np.load(
                os.path.join(save_path_flow, "flow_strength.npy")
            )
            flow_direction_x = np.load(
                os.path.join(save_path_flow, "flow_direction_x.npy")
            )
            flow_direction_y = np.load(
                os.path.join(save_path_flow, "flow_direction_y.npy")
            )
            print("Flow loaded")

        else:
            flow_strength = np.zeros((len(y), len(x)))
            flow_direction_x = np.zeros((len(y), len(x)))
            flow_direction_y = np.zeros((len(y), len(x)))

            for i in range(len(x)):
                for j in range(len(y)):
                    vx, vy = flow_field((x[i], y[j]))
                    magnitude = np.sqrt(vx**2 + vy**2)
                    if magnitude > 0:
                        flow_strength[j, i] = magnitude
                        flow_direction_x[j, i] = vx / magnitude
                        flow_direction_y[j, i] = vy / magnitude
            os.makedirs(save_path_flow, exist_ok=True)
            np.save(os.path.join(save_path_flow, "flow_strength.npy"), flow_strength)
            np.save(
                os.path.join(save_path_flow, "flow_direction_x.npy"), flow_direction_x
            )
            np.save(
                os.path.join(save_path_flow, "flow_direction_y.npy"), flow_direction_y
            )

        print("Flow computed")
        vx = flow_direction_x * flow_strength
        vy = flow_direction_y * flow_strength
    mask = np.ones_like(phi, dtype=bool)
    i_goal = np.argmin(np.abs(x - goal_point[0]))
    j_goal = np.argmin(np.abs(y - goal_point[1]))
    mask[j_goal, i_goal] = False

    # travel_time = skfmm.travel_time(mask, speed, dx=domain_size[0] / grid_size[0])
    v0 = speed
    travel_time = ordered_upwind_method_fast(x, y, v0, vx, vy, goal_point)
    travel_time = gaussian_filter(travel_time, sigma=1)

    path = []
    current = start_point
    path.append(current)

    travel_time_interp = RegularGridInterpolator(
        (y, x), travel_time, bounds_error=False, fill_value=None
    )

    step_size = min(domain_size) / 100
    max_iterations = 1000
    convergence_threshold = step_size / 2

    for _ in range(max_iterations):
        eps = min(domain_size) / 1000
        dx_points = [(current[0] + eps, current[1]), (current[0] - eps, current[1])]
        dy_points = [(current[0], current[1] + eps), (current[0], current[1] - eps)]

        dx_values = [
            travel_time_interp(p[::-1]) for p in dx_points
        ]  # Note: inversé car travel_time_interp attend (y, x)
        dy_values = [travel_time_interp(p[::-1]) for p in dy_points]

        gradient_x = (dx_values[0] - dx_values[1]) / (2 * eps)
        gradient_y = (dy_values[0] - dy_values[1]) / (2 * eps)

        # Normaliser le gradient
        gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
        if gradient_norm < 1e-10:
            break

        gradient_x /= gradient_norm
        gradient_y /= gradient_norm

        next_x = current[0] - step_size * gradient_x
        next_y = current[1] - step_size * gradient_y
        next_point = (next_x, next_y)

        distance_to_goal = np.sqrt(
            (next_point[0] - goal_point[0]) ** 2 + (next_point[1] - goal_point[1]) ** 2
        )
        if distance_to_goal < convergence_threshold:
            path.append(goal_point)
            break

        path.append(next_point)
        current = next_point

    return path, travel_time, (x, y), save_path_phi, save_path_flow


def visualize_results(
    path,
    travel_time,
    grid_info,
    sdf_function,
    flow_field=None,
):
    """
    Visualise le chemin trouvé, la carte de temps, la SDF et l'écoulement
    """
    x, y = grid_info
    X, Y = np.meshgrid(x, y)

    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            a = sdf_function((x[i], y[j]))
            phi[j, i] = a
    phi = phi / np.max(np.abs(phi))

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    travel_time = travel_time / np.std(travel_time)
    contour = ax[0].contourf(X, Y, travel_time, 50, cmap="viridis")
    fig.colorbar(contour, ax=ax[0], label="Travel time")

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax[0].plot(path_x, path_y, "r-", linewidth=2)
    ax[0].set_aspect("equal")
    ax[0].set_title("Map of travel time")

    contour_sdf = ax[1].contourf(X, Y, phi, levels=100, cmap="viridis")
    plt.colorbar(contour_sdf, label="Signed Distance")
    ax[1].contour(X, Y, phi, levels=[0], colors="k", linewidths=2)

    ax[1].plot(path_x, path_y, "r-", linewidth=2)
    ax[1].set_aspect("equal")
    ax[1].set_title("SDF with path")

    if flow_field is not None:
        flow_x = np.zeros((len(y), len(x)))
        flow_y = np.zeros((len(y), len(x)))

        for i in range(len(x)):
            for j in range(len(y)):
                if phi[j, i] < 0:
                    vx, vy = flow_field((x[i], y[j]))
                    flow_x[j, i] = vx
                    flow_y[j, i] = vy

        step = 10
        ax[1].quiver(
            X[::step, ::step],
            Y[::step, ::step],
            flow_x[::step, ::step],
            flow_y[::step, ::step],
            color="white",
            scale=30,
            alpha=0.7,
        )

    ax[0].plot(path_x[0], path_y[0], "ro", markersize=4)
    ax[0].plot(path_x[-1], path_y[-1], "ro", markersize=4)
    ax[1].plot(path_x[0], path_y[0], "ro", markersize=4)
    ax[1].plot(path_x[-1], path_y[-1], "ro", markersize=4)

    plt.tight_layout()

    return fig


def sdf_many_circle(point):
    centers = [(1 / 3, 1 / 3), (3 / 4, 3 / 4)]
    radius = 1 / 5
    min_distance = 0.1
    distances = np.zeros(len(centers))
    for i, center in enumerate(centers):
        distances[i] = np.linalg.norm(np.array(center) - np.array(point))

    id = np.argmin(distances)
    center = centers[id]
    return sdf_circle(point, center, radius)


def simple_flow(point):
    return (1.0, 1.0)


def plot_different_B():
    fig, ax = plt.subplots(figsize=(15, 7))
    start_point = (0.05, 0.5)
    goal_point = (0.95, 0.5)
    B_values = np.linspace(3, 10, 3, dtype="int")
    palette = sns.color_palette()
    grid_size = (200, 200)
    domain_size = (1.0, 1.0)
    x = np.linspace(0, domain_size[0], grid_size[0])
    y = np.linspace(0, domain_size[1], grid_size[1])
    for id, B in enumerate(B_values):
        path, travel_time, grid_info = compute_fmm_path(
            start_point, goal_point, sdf_many_circle, x, y, B, None
        )
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(
            path_x, path_y, "-", linewidth=2, color=palette[id + 2], label=f"B : {B}"
        )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    x, y = grid_info
    X, Y = np.meshgrid(x, y)
    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            phi[j, i] = sdf_many_circle((x[i], y[j]))
    contour_sdf = ax.contourf(X, Y, phi, levels=100, cmap="viridis")
    cbar = plt.colorbar(contour_sdf, ax=ax, pad=0.1, label="Signed Distance")
    cbar.ax.tick_params(labelsize=10)
    ax.contour(X, Y, phi, levels=[0], colors="k", linewidths=2)
    ax.set_title("SDF with path")
    plt.savefig("fig/fmm/fmm_B.png", dpi=200, bbox_inches="tight")
    plt.close()


def circle_path():
    fig, ax = plt.subplots(figsize=(15, 7))
    start_point = (0.05, 0.5)
    goal_point = (0.95, 0.5)
    domain_size = (1.0, 1.0)
    grid_size = (200, 200)
    x = np.linspace(0, domain_size[0], grid_size[0])
    y = np.linspace(0, domain_size[1], grid_size[1])
    B = 5
    palette = sns.color_palette()
    path, travel_time, grid_info = compute_fmm_path(
        start_point,
        goal_point,
        sdf_many_circle,
        x,
        y,
        B,
        None,
        grid_size=grid_size,
        domain_size=domain_size,
    )
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, "-", linewidth=2, color=palette[0], label=f"B : {B}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    x, y = grid_info
    X, Y = np.meshgrid(x, y)
    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            phi[j, i] = sdf_many_circle((x[i], y[j]))
    contour_sdf = ax.contourf(X, Y, phi, levels=100, cmap="coolwarm")
    cbar = plt.colorbar(contour_sdf, ax=ax, pad=0.1, label="Signed Distance")
    cbar.ax.tick_params(labelsize=10)
    ax.contour(X, Y, phi, levels=[0], colors="k", linewidths=2)
    ax.set_title("SDF with path")
    plt.savefig("fig/fmm/fmm_two_circles.png", dpi=200, bbox_inches="tight")
    plt.close()


def sdf_func_and_velocity_func(domain_size, ratio):
    x, y, N, h, sdf = load_sdf_from_csv(domain_size)
    sdf_interpolator = RegularGridInterpolator(
        (y, x), sdf, bounds_error=False, fill_value=None
    )

    def sdf_function(point):
        return sdf_interpolator(point[::-1])

    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)
    v = vel[N[2] // 2, :, :, 0:2]
    vx, vy = v[:, :, 0], v[:, :, 1]
    velocity_interpolator_x = RegularGridInterpolator(
        (y, x), vx, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y = RegularGridInterpolator(
        (y, x), vy, bounds_error=False, fill_value=None
    )
    v_magnitude = np.sqrt(vx**2 + vy**2)

    def velocity_retina(point):
        return (
            ratio
            * np.array(
                [
                    velocity_interpolator_x(point[::-1]),
                    velocity_interpolator_y(point[::-1]),
                ]
            )
            / np.max(v_magnitude)
        )

    return sdf_function, velocity_retina


def retina_path():
    scale = 8
    ratio = 1.5

    start_point = (0.98, 0.3)
    goal_point = (0.33, 0.5)
    domain_size = (1 * scale, 1 * scale)
    x, y, N, h, sdf = load_sdf_from_csv(domain_size)
    res_factor = 2
    grid_size = (N[0] * res_factor, N[1] * res_factor)
    start_point = (start_point[0] * scale, start_point[1] * scale)
    goal_point = (goal_point[0] * scale, goal_point[1] * scale)
    
    sdf_interpolator = RegularGridInterpolator(
        (y, x), sdf, bounds_error=False, fill_value=None
    )

    x_new = np.linspace(0, domain_size[0], grid_size[0])
    y_new = np.linspace(0, domain_size[1], grid_size[1])
    X_new, Y_new = np.meshgrid(x_new, y_new)
    sdf = sdf_interpolator((Y_new, X_new))

    sdf_function, velocity_retina = sdf_func_and_velocity_func(domain_size, ratio)
    B = 20
    flow_factor = 2
    path, travel_time, grid_info, save_path_phi, save_path_flow = compute_fmm_path(
        start_point,
        goal_point,
        sdf_function,
        x_new,
        y_new,
        B=B,
        flow_field=velocity_retina,
        grid_size=grid_size,
        domain_size=domain_size,
        ratio=ratio,
        flow_factor=flow_factor,
    )

    fig = visualize_results(
        path, travel_time, (x_new, y_new), sdf_function, flow_field=velocity_retina
    )
    plt.savefig("fig/fmm/fmm_retina_velocity_3_5.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    retina_path()

    # fig = visualize_results(path, travel_time, grid_info, sdf_many_circle, None)


if __name__ == "__main__":
    main()
