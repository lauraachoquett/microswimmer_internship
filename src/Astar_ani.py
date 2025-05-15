import heapq
import os
import time
from datetime import datetime
from math import ceil, sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator, splev, splprep
from scipy.ndimage import gaussian_filter1d

from src.Astar import resample_path
from src.data_loader import load_sdf_from_csv, vel_read,load_sim_sdf
from src.fmm import sdf_func_and_velocity_func
from src.sdf import get_contour_coordinates


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
    x,
    y,
    v0,
    vx,
    vy,
    start_point,
    goal_point,
    sdf_function,
    heuristic_weight=1.0,
    weight_sdf=1,
    pow_v0=1,
    pow_al=1
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
    move_costs = precalculate_move_costs(
        v0, vx, vy, dir_offsets, dx, dy, weight_sdf, pow_v0,pow_al
    )
    if sdf_function(goal_point)>0 or sdf_function(start_point)>0:
        print("Invalid points")
        print(sdf_function(goal_point))
        print(sdf_function(start_point))
        
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


def precalculate_move_costs(v0, vx, vy, dir_offsets, dx, dy, weight_sdf, pow_v0,pow_al):
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
                    if vx is None or vy is None:
                        move_costs[j, i, d_idx] = distance
                        continue
                    dir_x = di * dx / distance
                    dir_y = dj * dy / distance
                    d = np.array([dir_x, dir_y])
                    v_l = np.array([vx[j, i], vy[j, i]])
                    v = v_l + U * d
                    flow_component = v @ d
                    if np.linalg.norm(v_l) > 0.0001:
                        alignment = np.linalg.norm(v_l @ d) / (
                             np.linalg.norm(d) *  np.linalg.norm(v_l)
                        )
                    else:
                        alignment = 1
                    alignment = alignment ** (pow_al)
                    effective_speed = flow_component * alignment
                    effective_speed = v0[j, i] ** (pow_v0) * max(effective_speed, 0.001)
                    if pow_al > 0  or pow_v0 > 0:
                        if v_l @ d > 0 and v0[j, i] > 0:
                            move_costs[j, i, d_idx] = distance / effective_speed
                    else : 
                        if flow_component > 0:
                            move_costs[j, i, d_idx] = distance / (flow_component)
    return move_costs


def heuristic(i1, j1, i2, j2, dx, dy):
    """
    Calcule une heuristique admissible pour A*.
    Utilise la distance euclidienne divisée par la vitesse maximale possible.
    """
    distance = np.sqrt(((i2 - i1) * dx) ** 2 + ((j2 - j1) * dy) ** 2)
    v_max = 1.0
    return distance / v_max

def plot_velocity(step,vx,vy,v0,X,Y):
    step = 8

    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    vx_sub = vx[::step, ::step]
    vy_sub = vy[::step, ::step]
    v0_sub = v0[::step, ::step]
    print(v0_sub.shape)
    mask = ((vx_sub != 0) | (vy_sub != 0))  & (v0_sub>0.2)

    X_masked = X_sub[mask]
    Y_masked = Y_sub[mask]
    vx_masked = vx_sub[mask]
    vy_masked = vy_sub[mask]
    plt.imshow(
        v0,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        cmap='Reds',
        alpha=0.7
    )
    plt.colorbar(label='v0')

    plt.quiver(
        X_masked,
        Y_masked,
        vx_masked,
        vy_masked,
        color="darkred",
        scale=80,
        alpha=0.5,
    )

def visualize_results_a_star(
    X, Y, sdf_function, path, vx, vy,v0, scale,label = '',color=None
    ):
    """
    Visualise les résultats de l'algorithme A*.
    """


    obstacle_contour = contour_2D(sdf_function, X, Y, scale)

    plt.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.2)
    palette = sns.color_palette()

    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, linewidth=2, label=label)
    # plt.scatter([path[0][0]], [path[0][1]], s=50)
    # plt.scatter([path[-1][0]], [path[-1][1]], s=20, color = palette[id])
    

def compute_v(x, y, velocity_retina, B, grid_size, ratio, sdf_function, c):

    if len(x) != grid_size[0] and len(y) != grid_size[1]:
        raise ValueError("x,y are not coherent with the size of the grid")
    flow_field = velocity_retina
    save_path_phi = f"data/phi/grid_size_{grid_size[0]}_{grid_size[1]}_phi_bis.npy"
    if os.path.exists(save_path_phi):
        phi = np.load(save_path_phi)
    else:
        phi = np.zeros((grid_size[1], grid_size[0]))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                a = sdf_function((x[i], y[j]))
                phi[j, i] = a
        os.makedirs(os.path.dirname(save_path_phi), exist_ok=True)
        np.save(save_path_phi, phi)

    speed = (1.0 / (1.0 + np.exp(B * phi))) - c
    speed = np.clip(speed, 0.001, 1.0)


    
    save_path_flow = f"data/velocity_flow/grid_size_{grid_size[0]}_{grid_size[1]}_bis/"
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

        else:
            flow_strength = np.zeros((grid_size[1], grid_size[0]))
            flow_direction_x = np.zeros((grid_size[1], grid_size[0]))
            flow_direction_y = np.zeros((grid_size[1], grid_size[0]))

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
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

        print("Flow ready")
        vx = flow_direction_x * flow_strength
        vy = flow_direction_y * flow_strength
    return speed, vx, vy, save_path_phi, save_path_flow

def plot_different_path(files_path,label_list,x_new,y_new,sdf_function,vx,vy,v0,scale) : 
    path_list = []
    palette = sns.color_palette()
    
    for id,file_path in enumerate(files_path) : 
        path = np.load(file_path)
        visualize_results_a_star(
            x_new, y_new, sdf_function, path, vx, vy,v0,scale,label = label_list[id],color = palette[id]
        )

    plt.legend()
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Path")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_ani_test_{current_time}.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":

    ratio=5
    
    # Determine the size of the domain. It maps each point of the domain to each point on the grid.
    sdf_func,velocity_retina,x_phys,y_phys,physical_width,physical_height,scale= load_sim_sdf(ratio)

    # sdf_function :  Calculate the sdf in any point of the domain py interpolation
    # velocity_retina :  Calculate the velocity in any point of the domain py interpolation

    # Reduce the cell size by a factor : res_factor

    c = 0.4
    # Compute v0,vx and vy on this new domain.
    plt.figure(figsize=(12, 10))
    weight_sdf = 1
    start_point = (physical_width * 0.98, physical_height * 0.3)
    # goal_point = (goal_point[0] * scale, goal_point[1] * scale)
    # goal_point = (17.095518023108937/20,12.52076514689449/20)
    goal_point =  (5.762615626076424/20,16.142539758719423/20)
    # goal_point=(7.855210513776498,19.169570750237117)
    # goal_point = ( 8.670006951086492,10.962489435445624)
    goal_point = (physical_width * goal_point[0], physical_height * goal_point[1])
    print('distance between the two points : ',  np.linalg.norm(np.array(goal_point)-np.array(start_point)))
    B = 5
    h = 2
    pow_v0 = 4
    pow_al = 7


    shortest_geo_path = False
    v1 = False
    grid_size = (len(x_phys),len(y_phys))
    v0, vx, vy, _, _ = compute_v(
        x_phys, y_phys, velocity_retina, B, grid_size, ratio, sdf_func, c
    )
    print("Shape : ")
    print(v0.shape)
    print(vx.shape)
    print(vy.shape)
    if shortest_geo_path: 
        v0 = np.ones_like(v0)
        vx = None
        vy = None
        h = 0.1
    
    if v1 : 
        v0 = np.ones_like(v0)
        pow_v0 = 1 
        pow_al = 0
        h = 0.1
    
        
        
    start_time = time.time()
    # Compute the path
    path, travel_time = astar_anisotropic(
        x_phys,
        y_phys,
        v0,
        vx,
        vy,
        start_point,
        goal_point,
        sdf_func,
        heuristic_weight=h,
        pow_v0=pow_v0,
        pow_al=pow_al
    )
    # path = shortcut_path(path,is_collision_free,sdf_interpolator)
    print("Travel time :", travel_time)

    path = np.array(path)  # de forme (N, 2)
    dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
    print("path before resampling :", len(path))
    n = ceil(np.max(dist) / (5 * 1e-3))
    if n > 1:
        path,distances = resample_path(path, len(path) * n)
    print("after resampling : ", len(path))
    smoothed_x = gaussian_filter1d(path[:, 0], sigma=30)
    smoothed_y = gaussian_filter1d(path[:, 1], sigma=30)
    path = np.stack([smoothed_x, smoothed_y], axis=1)
    print("Path length : ",distances)
    X, Y = np.meshgrid(x_phys, y_phys)
    visualize_results_a_star(
        X, Y, sdf_func, path, vx, vy,v0,scale,label = 'path'
    )
    plot_velocity(6,vx,vy,v0,X,Y)
    plt.legend()
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Path")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_ani_test_{current_time}.png", dpi=300, bbox_inches="tight")
    plt.close()

    
    save_path_path = f"data/retina2D_path_time_4_v1_bg.npy"
    np.save(save_path_path, path, allow_pickle=False)

    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("Execution time:", elapsed_time, "minutes")
    # files_path = ['data/retina2D_path_time_4_free_bg.npy','data/retina2D_path_time_4_v1_bg.npy','data/retina2D_path_time_4_v2_bg.npy']
    # label_list = ['Shortest geometrical path','Algorithm 1', 'Algorithm 2']
    # plot_different_path(files_path,label_list,x_phys,y_phys,sdf_func,vx,vy,v0,scale)

        
    
