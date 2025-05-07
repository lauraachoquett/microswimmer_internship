import heapq
import os
import time
from datetime import datetime
from math import ceil
import numpy as np
from scipy.interpolate import RegularGridInterpolator, splev, splprep
from scipy.ndimage import gaussian_filter1d

from .Astar import resample_path
from .data_loader import load_sdf_from_csv, vel_read
from .fmm import sdf_func_and_velocity_func
from .sdf import get_contour_coordinates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def contour_2D(sdf_function, X_new, Y_new, scale):
    if os.path.exists(f"data/retina2D_contour_scale_{scale}.npy"):
        obstacle_contour = np.load(
            f"data/retina2D_contour_scale_{scale}.npy", allow_pickle=False
        )
        print("Contour loaded")
    else:
        Z = np.vectorize(lambda px, py: sdf_function((px, py)))(X_new, Y_new)
        obstacle_contour = get_contour_coordinates(X_new, Y_new, Z, level=0)
        np.save(
            f"data/retina2D_contour_scale_{scale}.npy",
            obstacle_contour,
            allow_pickle=False,
        )

    return obstacle_contour


def astar_anisotropic(
    x, y, v0, vx, vy, start_point, goal_point, sdf_function, heuristic_weight=1.0
):
    """
    Implémentation de l'algorithme A* adapté pour les écoulements anisotropes.

    Paramètres:
    - x, y : tableaux 1D des coordonnées de la grille
    - v0 : vitesse propre du milieu, tableau 2D de taille (len(y), len(x))
    - vx, vy : composantes du champ de fluide, tableaux 2D de taille (len(y), len(x))
    - start_point : tuple (x_start, y_start) représentant le point de départ
    - goal_point : tuple (x_goal, y_goal) représentant le point d'arrivée
    - heuristic_weight : poids de l'heuristique (>=0.0, où 0.0 donne un A* optimal)
    - directions : nombre de directions possibles (8, 16, 32...)

    Retourne:
    - path : liste de points (x, y) représentant le chemin optimal
    - travel_time : tableau 2D des temps de trajet pour chaque point visité
    """
    nx, ny = len(x), len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    i_start = np.argmin(np.abs(x - start_point[0]))
    j_start = np.argmin(np.abs(y - start_point[1]))
    i_goal = np.argmin(np.abs(x - goal_point[0]))
    j_goal = np.argmin(np.abs(y - goal_point[1]))

    dir_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    move_costs = precalculate_move_costs(v0, vx, vy, dir_offsets, dx, dy)

    closed_set = set()
    open_set = [(0, i_start, j_start)]
    heapq.heapify(open_set)

    came_from = {}
    g_score = np.full((ny, nx), np.inf)
    g_score[j_start, i_start] = 0

    f_score = np.full((ny, nx), np.inf)
    h_goal = heuristic(i_start, j_start, i_goal, j_goal, dx, dy)
    f_score[j_start, i_start] = h_goal

    travel_time = np.full((ny, nx), np.inf)
    travel_time[j_start, i_start] = 0

    while open_set:

        f, current_i, current_j = heapq.heappop(open_set)
        # print("f current point :",f)
        if current_i == i_goal and current_j == j_goal:

            path = [(x[i_goal], y[j_goal])]
            i, j = i_goal, j_goal
            while (i, j) in came_from:
                i, j = came_from[(i, j)]
                path.append((x[i], y[j]))
            path.reverse()
            return path, travel_time[j_goal, i_goal]

        closed_set.add((current_i, current_j))

        for di, dj in dir_offsets:
            neighbor_i, neighbor_j = current_i + di, current_j + dj

            if not (0 <= neighbor_i < nx and 0 <= neighbor_j < ny):
                continue
            if sdf_function((x[neighbor_i], y[neighbor_j])) > 0:
                continue

            if (neighbor_i, neighbor_j) in closed_set:
                continue

            cost = move_costs[current_j, current_i, dir_offsets.index((di, dj))]
            if cost == np.inf:
                continue

            tentative_g_score = g_score[current_j, current_i] + cost

            if tentative_g_score >= g_score[neighbor_j, neighbor_i]:
                continue

            came_from[(neighbor_i, neighbor_j)] = (current_i, current_j)
            g_score[neighbor_j, neighbor_i] = tentative_g_score
            travel_time[neighbor_j, neighbor_i] = tentative_g_score

            h = heuristic(neighbor_i, neighbor_j, i_goal, j_goal, dx, dy)
            f_score[neighbor_j, neighbor_i] = tentative_g_score + heuristic_weight * h

            heapq.heappush(
                open_set, (f_score[neighbor_j, neighbor_i], neighbor_i, neighbor_j)
            )

    return [], travel_time


def precalculate_move_costs(v0, vx, vy, dir_offsets, dx, dy):
    """
    Pré-calcule les coûts de déplacement pour chaque direction à chaque point.

    Retourne:
    - move_costs: tableau 3D (ny, nx, n_directions) des coûts
    """
    ny, nx = v0.shape
    n_directions = len(dir_offsets)
    move_costs = np.full((ny, nx, n_directions), np.inf)

    U = 1

    for j in range(ny):
        for i in range(nx):
            for d_idx, (di, dj) in enumerate(dir_offsets):
                distance = np.sqrt((di * dx) ** 2 + (dj * dy) ** 2)

                if distance > 0:
                    dir_x = di * dx / distance
                    dir_y = dj * dy / distance
                    d = np.array([dir_x, dir_y])
                    v_l = np.array([vx[j, i], vy[j, i]]) + U * d
                    v = v_l + U * d
                    flow_component = v @ d
                    alignment = np.linalg.norm(np.cross(v, d)) / (
                        np.linalg.norm(v) * np.linalg.norm(d)
                    )
                    alignment = max(alignment, 0.1)
                    effective_speed = flow_component / alignment
                    effective_speed = v0[j, i] * max(effective_speed, 0.001)
                    if effective_speed > 0:
                        move_costs[j, i, d_idx] = distance / effective_speed

    return move_costs


def heuristic(i1, j1, i2, j2, dx, dy):
    """
    Calcule une heuristique admissible pour A*.
    Utilise la distance euclidienne divisée par la vitesse maximale possible.
    """
    distance = np.sqrt(((i2 - i1) * dx) ** 2 + ((j2 - j1) * dy) ** 2)
    v_max = 1.0
    return distance / v_max


def visualize_results_a_star(x, y, sdf_function, path, vx, vy, scale,B=None):
    """
    Visualise les résultats de l'algorithme A*.
    """


    # phi = np.zeros((len(y), len(x)))
    # for i in range(len(x)):
    #     for j in range(len(y)):
    #         a = sdf_function((x[i], y[j]))
    #         phi[j, i] = a
    # phi = phi / np.max(np.abs(phi))


    X, Y = np.meshgrid(x, y)

    obstacle_contour = contour_2D(sdf_function, X, Y, scale)

    plt.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.5)

    if len(path) > 0:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, linewidth=2,label=f'B : {B}')

        plt.scatter([path[0][0]], [path[0][1]], s=50)
        plt.scatter([path[-1][0]], [path[-1][1]], s=50)
    # step = 40

    # X_sub = X[::step, ::step]
    # Y_sub = Y[::step, ::step]
    # vx_sub = vx[::step, ::step]
    # vy_sub = vy[::step, ::step]

    # mask = (vx_sub != 0) | (vy_sub != 0)

    # X_masked = X_sub[mask]
    # Y_masked = Y_sub[mask]
    # vx_masked = vx_sub[mask]
    # vy_masked = vy_sub[mask]

    # plt.quiver(
    #     X_masked,
    #     Y_masked,
    #     vx_masked,
    #     vy_masked,
    #     color="darkred",
    #     scale=100,
    #     alpha=0.7,
    # )




def compute_v(x, y, velocity_retina, B, grid_size, ratio, sdf_function):

    if len(x) != grid_size[0] and len(y) != grid_size[1]:
        raise ValueError("x,y are not coherent with the size of the grid")
    flow_field = velocity_retina
    save_path_phi = f"data/phi/grid_size_{grid_size[0]}_{grid_size[1]}_phi.npy"
    if os.path.exists(save_path_phi):
        phi = np.load(save_path_phi)
        print("Phi loaded")
    else:
        phi = np.zeros((grid_size[1], grid_size[0]))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                a = sdf_function((x[i], y[j]))
                phi[j, i] = a
        phi = 3 * phi / np.max(np.abs(phi))
        os.makedirs(os.path.dirname(save_path_phi), exist_ok=True)
        np.save(save_path_phi, phi)
    print("SDF computed")
    speed = (1.0 / (1.0 + np.exp(B * phi))) - 1 / 2
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
            flow_strength = np.zeros((grid_size[1], grid_size[0]))
            flow_direction_x = np.zeros((grid_size[1], grid_size[0]))
            flow_direction_y = np.zeros((grid_size[1], grid_size[0]))

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    vx, vy = flow_field((x[i], y[j])) / ratio
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

    return speed, vx, vy, save_path_phi, save_path_flow


def shortcut_path(path, is_collision_free, sdf):
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i:
            if is_collision_free(path[i], path[j], sdf):
                smoothed.append(path[j])
                i = j
                break
            j -= 1
    return smoothed


def is_collision_free(p1, p2, sdf_interpolator):
    points = np.linspace(p1, p2, 20)
    try:
        values = sdf_interpolator(points)
    except ValueError:
        # Si les points sont hors des limites du domaine
        return False
    return np.all(values <= 0)


if __name__ == "__main__":
    start_time = time.time()

    scale = 20
    ratio = 5
    start_point = (0.98, 0.3)
    goal_point = (0.56, 0.08)
    domain_size = (1 * scale, 1 * scale)

    # Determine the size of the domain. It maps each point of the domain to each point on the grid.
    x, y, N, h, sdf = load_sdf_from_csv(domain_size)

    # Define sdf and velocity interpolator regarding the size of the domain. x,y and sdf must have the same size
    sdf_function, velocity_retina = sdf_func_and_velocity_func(domain_size, ratio)

    # sdf_function :  Calculate the sdf in any point of the domain py interpolation
    # velocity_retina :  Calculate the velocity in any point of the domain py interpolation

    # Reduce the cell size by a factor : res_factor
    res_factor = 1
    grid_size = (N[0] * res_factor, N[1] * res_factor)
    x_new = np.linspace(0, domain_size[0], grid_size[0])
    y_new = np.linspace(0, domain_size[1], grid_size[1])

    B = 1
    B_values = np.linspace(1/10,5,2)
    # Compute v0,vx and vy on this new domain.
    plt.figure(figsize=(12, 10))
    
    for B in B_values:
        v0, vx, vy, _, _ = compute_v(
            x_new, y_new, velocity_retina, B, grid_size, ratio, sdf_function
        )
        weight_sdf = 8
        start_point = (start_point[0] * scale, start_point[1] * scale)
        goal_point = (goal_point[0] * scale, goal_point[1] * scale)

        ## Compute the path
        path, travel_time = astar_anisotropic(
            x_new,
            y_new,
            v0,
            vx,
            vy,
            start_point,
            goal_point,
            sdf_function,
            heuristic_weight=0.5,
        )
        # path = shortcut_path(path,is_collision_free,sdf_interpolator)
        print("Travel time :", travel_time)
    
        path = np.array(path)  # de forme (N, 2)
        dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
        print("path before resampling :", len(path))
        n = ceil(np.max(dist) / (5 * 1e-3))
        if n > 1:
            path = resample_path(path, len(path) * n)
        print("after resampling : ",len(path))
        smoothed_x = gaussian_filter1d(path[:, 0], sigma=20)
        smoothed_y = gaussian_filter1d(path[:, 1], sigma=20)
        path = np.stack([smoothed_x, smoothed_y], axis=1)
        print("smoooooth")
        visualize_results_a_star(x_new, y_new, sdf_function, path, vx, vy, scale,B)


    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("Execution time:", elapsed_time, "minutes")
    
    plt.legend()
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("SDF with path")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_ani_test_{current_time}.png", dpi=300, bbox_inches="tight")
