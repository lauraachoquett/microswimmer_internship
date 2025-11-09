import heapq
import os
import time
from datetime import datetime
from math import ceil, sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator, splev, splprep
from scipy.ndimage import gaussian_filter1d

from src.Astar import resample_path
from src.data_loader import load_sdf_from_csv, vel_read,load_sim_sdf
from src.fmm import sdf_func_and_velocity_func
from src.sdf import get_contour_coordinates

from math import gcd, sqrt
from matplotlib import cm
from matplotlib.colors import Normalize
import zipfile


plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,  # texte vectoriel dans les PDF
    "ps.fonttype": 42,   # aussi pour EPS
})

def generate_directions(max_radius):
    directions = set()
    for dx in range(-max_radius, max_radius + 1):
        for dy in range(-max_radius, max_radius + 1):
            if dx == 0 and dy == 0:
                continue
            distance = sqrt(dx**2 + dy**2)
            if distance <= max_radius:
                # Réduction de la direction (évite les doublons comme (2,2) si (1,1) est déjà là)
                g = gcd(abs(dx), abs(dy))
                reduced = (dx // g, dy // g)
                directions.add(reduced)
    return list(directions)


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
    pow_al=1,
    max_radius=2,
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

    dir_offsets = generate_directions(max_radius)
    print("Number of directions : ", len(dir_offsets))
    move_costs = precalculate_move_costs(
        v0, vx, vy, dir_offsets, dx, dy, weight_sdf, pow_v0,pow_al
    )
    print("Cost computed")
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

        for d_idx,(di, dj) in enumerate(dir_offsets):
            neighbor_i, neighbor_j = current_i + di, current_j + dj

            if not (0 <= neighbor_i < nx and 0 <= neighbor_j < ny):
                continue
            if sdf_function((x[neighbor_i], y[neighbor_j])) > 0:
                continue

            if (neighbor_i, neighbor_j) in closed_set:
                continue

            cost = move_costs[current_j, current_i, d_idx]
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



def precalculate_move_costs(v0, vx, vy, dir_offsets, dx, dy, weight_sdf, pow_v0, pow_al):
    """
    Pré-calcule les coûts de déplacement pour chaque direction à chaque point.
    Version vectorisée optimisée.

    Retourne:
    - move_costs: tableau 3D (ny, nx, n_directions) des coûts
    """
    ny, nx = v0.shape
    n_directions = len(dir_offsets)
    
    dir_offsets = np.array(dir_offsets)  # shape: (n_directions, 2)
    
    distances = np.sqrt((dir_offsets[:, 0] * dx) ** 2 + (dir_offsets[:, 1] * dy) ** 2)
    
    valid_dirs = distances > 0

    move_costs = np.full((ny, nx, n_directions), np.inf)
    
    if vx is None or vy is None:
        move_costs[:, :, valid_dirs] = distances[valid_dirs]
        return move_costs
    
    U = 1
    
    dir_norm = np.zeros((n_directions, 2))
    dir_norm[valid_dirs, 0] = dir_offsets[valid_dirs, 0] * dx / distances[valid_dirs]
    dir_norm[valid_dirs, 1] = dir_offsets[valid_dirs, 1] * dy / distances[valid_dirs]
    
    for d_idx in np.where(valid_dirs)[0]:
        di, dj = dir_offsets[d_idx]
        distance = distances[d_idx]
        dir_x, dir_y = dir_norm[d_idx]
        
        neighbor_i = np.clip(np.arange(nx)[None, :] + di, 0, nx-1)
        neighbor_j = np.clip(np.arange(ny)[:, None] + dj, 0, ny-1)
        
        # Champ de vitesse local en chaque point
        v_l_x = vx  # shape: (ny, nx)
        v_l_y = vy  # shape: (ny, nx)
        
        # Vitesse totale v = v_l + U * d
        v_x = v_l_x + U * dir_x
        v_y = v_l_y + U * dir_y
        
        # Composante du flux dans la direction
        flow_component = v_x * dir_x + v_y * dir_y
        
        # Calcul de l'alignement
        v_l_norm = np.sqrt(v_l_x**2 + v_l_y**2)
        v_l_dot_d = v_l_x * dir_x + v_l_y * dir_y
        
        # Éviter division par zéro
        alignment = np.ones_like(v_l_norm)
        mask_nonzero = v_l_norm > 0.0001
        alignment[mask_nonzero] = (np.abs(v_l_dot_d[mask_nonzero]) / v_l_norm[mask_nonzero]) ** pow_al
        
        # Vitesse effective
        effective_speed = flow_component * alignment
        
        # Facteur de vitesse du voisin
        v0_neighbor = v0[neighbor_j, neighbor_i]
        effective_speed = (v0_neighbor ** pow_v0) * np.maximum(effective_speed, 0.001)
        
        # Conditions pour calculer le coût
        if pow_al > 0 or pow_v0 > 0:
            valid_mask = (v_l_dot_d > 0.0)
        else:
            valid_mask = flow_component > 0
        
        # Calculer les coûts
        costs = np.full((ny, nx), np.inf)
        if pow_al > 0 or pow_v0 > 0:
            costs[valid_mask] = distance / effective_speed[valid_mask]
        else:
            costs[valid_mask] = distance / flow_component[valid_mask]
        
        move_costs[:, :, d_idx] = costs
    
    return move_costs


def heuristic(i1, j1, i2, j2, dx, dy):
    """
    Calcule une heuristique admissible pour A*.
    Utilise la distance euclidienne divisée par la vitesse maximale possible.
    """
    distance = np.sqrt(((i2 - i1) * dx) ** 2 + ((j2 - j1) * dy) ** 2)
    v_max = 1.0
    return distance / v_max


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
    print("Shape of phi :",phi.shape)
    print("X :",len(x))
    print("Y :",len(y))
    speed = (1.0 / (1.0 + np.exp(B * phi))) - c


    
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
        vx = flow_direction_x * flow_strength / (np.max(flow_strength)) * ratio
        vy = flow_direction_y * flow_strength / (np.max(flow_strength)) * ratio
        print("vx shape :",vx.shape)
        print("vy shape :",vy.shape)
    return speed, vx, vy, save_path_phi, save_path_flow

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter

def plot_velocity_quiver(step, vx, vy, v0, X, Y, grid_size):
    # Sous-échantillonnage
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    vx_sub = vx[::step, ::step]
    vy_sub = vy[::step, ::step]
    v0_sub = v0[::step, ::step]

    # Norme du champ
    v_norm = np.sqrt(vx_sub**2 + vy_sub**2)
    print("Max velocity magnitude :", np.max(v_norm))

    # 1. Masque de base : vitesse non nulle + fluide actif
    mask = ((vx_sub != 0) | (vy_sub != 0)) & (v0_sub > 0.08)

    # 2. Supprimer les vecteurs trop petits (parasites visuellement)
    min_magnitude = 1e-1
    mask &= v_norm > min_magnitude

    # 3. Détection d’outliers locaux
    local_median = median_filter(v_norm, size=3)
    mask &= v_norm <= 2 * local_median

    # Extraction des valeurs filtrées
    X_masked = X_sub[mask]
    Y_masked = Y_sub[mask]
    vx_masked = vx_sub[mask]
    vy_masked = vy_sub[mask]
    arrow_colors = v_norm[mask]

    # Quiver plot
    quiv = plt.quiver(
        X_masked, Y_masked,
        vx_masked, vy_masked,
        arrow_colors,
        cmap="Reds",
        scale=70,
        alpha=0.9,
        zorder=2
    )

    # Normalisation couleur et barre
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=np.percentile(arrow_colors, 5), vmax=arrow_colors.max())
    quiv.set_cmap(cmap)
    quiv.set_norm(norm)
    plt.colorbar(quiv, label=r'$||v||/U$ (arrows)')

def plot_velocity_map(v_filtered, X, Y, vmin, vmax):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("Reds")

    # Affichage
    im = plt.pcolormesh(X, Y, v_filtered, shading='auto', cmap=cmap, norm=norm,rasterized=True)
    cbar = plt.colorbar(im, label=r'$||v||/U$ (magnitude)')
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')
    return im

def visualize_results_a_star(
    X, Y, sdf_function, path, scale, label=None, color=None
):

    obstacle_contour = contour_2D(sdf_function, X, Y, scale)
    plt.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.5,rasterized=True)

    path_x, path_y = zip(*path)
    smoothed_x = gaussian_filter1d(path_x, sigma=15)
    smoothed_y = gaussian_filter1d(path_y, sigma=15)
    plt.plot(smoothed_x, smoothed_y, linewidth=3, label=label, color=color)

    
def create_vel_different_path(
    vx: np.ndarray,
    vy: np.ndarray,
    v0: np.ndarray,
    dir: str,
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    *,
    mask_threshold: float = 0.08,
    median_size: int = 3,
):
    os.makedirs(dir, exist_ok=True)
    os.makedirs(dir, exist_ok=True)

    X, Y = np.meshgrid(x_phys, y_phys, indexing="xy")
    np.savez(os.path.join(dir, f"grid.npz"), X=X, Y=Y)

    v_norm = np.hypot(vx, vy).astype(float)

    mask_domain = v0 > mask_threshold

    local_median = median_filter(v_norm, size=median_size)
    outlier_mask = v_norm > 2.0 * np.clip(local_median, 1e-12, None)

    v_filtered = v_norm.copy()
    v_filtered[outlier_mask] = np.nan
    v_filtered[~mask_domain] = np.nan

    v_filtered_fp = os.path.join(dir, f"v_filtered.npy")
    np.save(v_filtered_fp, v_filtered)

    return v_filtered_fp, os.path.join(dir, f"grid.npz")


def load_data(
    dir: str,
    contour_path:str,
    files_path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    v_percentile: float = 5.0,
):

    # Contour
    obstacle_contour = np.load(contour_path, allow_pickle=False)
    print(obstacle_contour.shape)
    grid_npz = os.path.join(dir, f"grid.npz")

    # Grid 
    with np.load(grid_npz) as d:
        X = d["X"]
        Y = d["Y"]

    # Velocity
    v_filtered_fp = os.path.join(dir, f"v_filtered.npy")
    v_filtered = np.load(v_filtered_fp, allow_pickle=False)

    if vmin is None:
        vmin = np.nanpercentile(v_filtered, v_percentile) if np.isfinite(v_filtered).any() else 0.0
    if vmax is None:
        vmax = np.nanmax(v_filtered) if np.isfinite(v_filtered).any() else 1.0
        
    paths=[]
    for file in files_path:
        path = np.load(file)
        path,_ = resample_path(path, len(path))
        paths.append(path)

    return X, Y, obstacle_contour, v_filtered, vmin, vmax,paths

def plot_different_path(paths,label_list,obstacle_contour,v_filtered,X,Y,vmin,vmax,sigma=15) : 
    palette = ['darkgreen','navy','cornflowerblue']
    im = plot_velocity_map(v_filtered,X,Y,vmin,vmax)
    
    for id,path in enumerate(paths) : 

        path_x, path_y = zip(*path)
        smoothed_x = gaussian_filter1d(path_x, sigma=sigma)
        smoothed_y = gaussian_filter1d(path_y, sigma=sigma)
        plt.plot(smoothed_x, smoothed_y, linewidth=3, label=label_list[id], color=palette[id])
            
    plt.scatter([path[0][0]], [path[0][1]], s=70,color=palette[1],label = 'Start',zorder=10)
    plt.scatter([path[-1][0]], [path[-1][1]], s=70, color = 'red',label = 'Target',zorder=10)
    plt.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.5,rasterized=True)
    
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_{current_time}.png", dpi=400, bbox_inches="tight")
    plt.savefig(f"fig/Astar_{current_time}.pdf", dpi=400, bbox_inches="tight")



if __name__ == "__main__":

    ratio=5
    
    # Determine the size of the domain. It maps each point of the domain to each point on the grid.
    sdf_func,velocity_retina,x_phys,y_phys,physical_width,physical_height,scale= load_sim_sdf(ratio)

    # sdf_function :  Calculate the sdf in any point of the domain py interpolation
    # velocity_retina :  Calculate the velocity in any point of the domain py interpolation

    # Reduce the cell size by a factor : res_factor
            
    # Compute v0,vx and vy on this new domain.
               
            
    plt.figure(figsize=(12, 10))
    # start_point = (float(physical_width * 0.98), float(physical_height * 0.3))
    # # goal_point_tmp = ( 17.38728683339246/20, 12.3556761323975/20)
    # # goal_point = (float(physical_width * goal_point_tmp[0]), float(physical_height * goal_point_tmp[1]))
    
    # goal_point = (12.991832733154297,13.56390380859375)
    # print('distance between the two points : ',  np.linalg.norm(np.array(goal_point)-np.array(start_point)))

    c = 0.4
    B = 1.6
    h = 0
    pow_v0 = 0
    pow_al = 5
    max_radius = 2

    shortest_geo_path = False
    v1 = False
    grid_size = (len(x_phys),len(y_phys))

    # current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    # save_dir = f"fig/Astar_ani_test_{current_time}"
    # os.makedirs(save_dir, exist_ok=True)


    # params = {
    #     "start_point": tuple(float(x) for x in start_point),
    #     "goal_point": tuple(float(x) for x in goal_point),
    #     "c": float(c),
    #     "B": float(B),
    #     "h": float(h),
    #     "pow_v0": float(pow_v0),
    #     "pow_al": float(pow_al),
    #     "max_radius": int(max_radius),
    #     "shortest_geo_path": bool(shortest_geo_path),
    #     "v1": bool(v1),
    #     "grid_size": tuple(int(x) for x in grid_size),
    #     "ratio": float(ratio),
    #     "scale": float(scale),
    # }
    # with open(os.path.join(save_dir, "params.json"), "w") as f:
    #     json.dump(params, f, indent=4)

    # for pow_v0 in np.linspace(0,7,7,dtype=int):
    v0, vx, vy, _, _ = compute_v(
        x_phys, y_phys, velocity_retina, B, grid_size, ratio, sdf_func, c
    )
    #     speed = np.sqrt(vx**2 + vy**2)
    #     # print(f"Speed: min={speed.min():.3e}, max={speed.max():.3e}, mean={speed.mean():.3e}, std={speed.std():.3e}")
        
    #     # for name, arr in zip(["vx", "vy","v0"], [vx, vy,v0]):
    #     #     print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")
        
    #     if shortest_geo_path: 
    #         v0 = np.ones_like(v0)
    #         vx = None
    #         vy = None
    #         h = 0.0
    #         max_radius=5
        
    #     if v1 : 
    #         v0 = np.ones_like(v0)
    #         pow_v0 = 0
    #         pow_al = 0
    #         h = 0.0
    #         max_radius= 1
        
            
    #     start_time = time.time()
    #     # Compute the path
    #     path, travel_time = astar_anisotropic(
    #         x_phys,
    #         y_phys,
    #         v0,
    #         vx,
    #         vy,
    #         start_point,
    #         goal_point,
    #         sdf_func,
    #         heuristic_weight=h,
    #         pow_v0=pow_v0,
    #         pow_al=pow_al,
    #         max_radius=max_radius
    #     )
    #     # path = shortcut_path(path,is_collision_free,sdf_interpolator)
    #     # print("Travel time :", travel_time)

    #     path = np.array(path)  # de forme (N, 2)
    #     dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
    #     # print("path before resampling :", len(path))
    #     n = ceil(np.max(dist) / (5 * 1e-3))
    #     if n > 1:
    #         path,distances = resample_path(path, len(path) * n)
    #     # print("after resampling : ", len(path))
    #     smoothed_x = gaussian_filter1d(path[:, 0], sigma=30)
    #     smoothed_y = gaussian_filter1d(path[:, 1], sigma=30)
    #     path = np.stack([smoothed_x, smoothed_y], axis=1)
    #     print("Path length : ",distances)
    #     X, Y = np.meshgrid(x_phys, y_phys)
    #     visualize_results_a_star(
    #         X, Y, sdf_func, path, vx, vy, v0, scale, label=f'a : {pow_v0:.2f}'
    #     )
    #     end_time = time.time()
    #     elapsed_time = (end_time - start_time) / 60
    #     print("Execution time:", elapsed_time, "minutes")
            
    # # if vx is not None and vy is not None : 
    # #     plot_velocity(6,vx,vy,v0,X,Y,grid_size)
    # plt.legend()
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.axis("off")
    # plt.title("Path")
    # plt.tight_layout()
    # path_save_fig = os.path.join(save_dir,f"Astar_ani_test_{current_time}_pow_v0.png")
    # plt.savefig(path_save_fig, dpi=300, bbox_inches="tight")
    # plt.close()

    
    # save_path_path = f"data/retina2D_path_time_4_v1_bg.npy"
    # np.save(save_path_path, path, allow_pickle=False)

    
    # dir = './data_a_star'
    # create_vel_different_path(vx,vy,v0,dir,x_phys,y_phys)
    
    zip_path = "data_a_star.zip"
    extract_dir = "./"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
            
    dir = './data_a_star'    
    contour_path = f'{dir}/retina2D_contour_scale_0.041784223169088364.npy'
    files_path = [f'{dir}/retina2D_path_time_4_free_bg.npy',f'{dir}/retina2D_path_time_4_v1_bg.npy',f'{dir}/retina2D_path_time_4_v2_bg.npy']
    
    X,Y,obstacle_contour,v_filtered,vmin,vmax ,paths= load_data(dir,contour_path,files_path)
    
    label_list = ['Shortest geometrical path','Algorithm 1', 'Algorithm 2']
    plot_different_path(paths,label_list,obstacle_contour,v_filtered,X,Y,vmin,vmax,sigma=15)

        
    
