import heapq
import os
import time
from datetime import datetime
from math import ceil, gcd, sqrt
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator, splev, splprep
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pyvista as pv
import numpy as np

from src.Astar import resample_path
from src.data_loader import load_sdf_from_csv, load_sim_sdf, vel_read,plot_sdf_slices
from src.fmm import sdf_func_and_velocity_func
from src.sdf import get_contour_coordinates

def gcd_of_three(a, b, c):
    return reduce(gcd, [abs(a), abs(b), abs(c)])

def generate_directions_3d(max_radius):
    directions = set()
    max_radius_int = ceil(max_radius)
    for dx in range(-max_radius_int, max_radius_int + 1):
        for dy in range(-max_radius_int, max_radius_int + 1):
            for dz in range(-max_radius_int, max_radius_int + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                
                distance = sqrt(dx**2 + dy**2 + dz**2)
                if distance <= max_radius:
                    if dx == 0 and dy == 0:
                        g = abs(dz)
                    elif dx == 0 and dz == 0:
                        g = abs(dy)
                    elif dy == 0 and dz == 0:
                        g = abs(dx)
                    else:
                        g = gcd_of_three(dx, dy, dz)
                    
                    if g > 0:
                        reduced = (dx // g, dy // g, dz // g)
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
    z,
    v0,
    vx,
    vy,
    vz,
    start_point,
    goal_point,
    sdf_function,
    sdf_phys,
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
    nx, ny,nz = len(x), len(y),len(z)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    X, Y,Z = np.meshgrid(x, y,z)
    i_start = np.argmin(np.abs(x - start_point[0]))
    j_start = np.argmin(np.abs(y - start_point[1]))
    k_start = np.argmin(np.abs(z - start_point[2]))
    i_goal = np.argmin(np.abs(x - goal_point[0]))
    j_goal = np.argmin(np.abs(y - goal_point[1]))
    k_goal = np.argmin(np.abs(z - goal_point[2]))
    
    plot_sdf_slices(sdf_phys,X,Y,Z,i_goal,j_goal,k_goal,goal_point,start_point,'slices_a_star')

    dir_offsets = generate_directions_3d(max_radius=max_radius)
    print("Number of different directions :", dir_offsets)
    
    if sdf_function(goal_point) > 0 or sdf_function(start_point) > 0:
        raise ValueError(
            f"Invalid points: sdf(goal_point)={sdf_function(goal_point)}, sdf(start_point)={sdf_function(start_point)}"
        )
    if v0[k_goal,j_goal,i_goal]<0:
        raise ValueError(
            f"Invalid points: v0(goal_point)={v0[k_goal,j_goal,i_goal]}"
        )
        
    move_costs = precalculate_move_costs(
        v0, vx, vy,vz, dir_offsets, dx, dy, dz,weight_sdf, pow_v0, pow_al
    )
    print("Cost computed, move shape :",move_costs.shape)

    closed_set = set()
    open_set = [(0, i_start, j_start,k_start)]
    heapq.heapify(open_set)

    came_from = {}
    g_score = np.full((nz,ny, nx), np.inf)
    g_score[k_start,j_start, i_start] = 0

    f_score = np.full((nz,ny, nx), np.inf)
    h_goal = heuristic_weight * heuristic(i_start, j_start,k_start, i_goal, j_goal,k_goal, dx, dy,dz)
    f_score[k_start,j_start, i_start] = h_goal

    travel_time = np.full((nz,ny, nx), np.inf)
    travel_time[k_start,j_start, i_start] = 0

    while open_set:

        f, current_i, current_j,current_k = heapq.heappop(open_set)
        # print("f current point :",f)
        if current_i == i_goal and current_j == j_goal and current_k == k_goal:

            path = [(x[i_goal], y[j_goal],z[k_goal])]
            i, j,k = i_goal, j_goal,k_goal
            while (i, j,k) in came_from:
                i, j,k = came_from[(i, j,k)]
                path.append((x[i], y[j],z[k]))
                print("V0 : ", v0[k,j,i]**pow_v0)
            path.reverse()
            return path, travel_time[k_goal,j_goal, i_goal]

        closed_set.add((current_i, current_j,current_k))

        for d_idx, (di, dj, dk) in enumerate(dir_offsets):
            neighbor_i, neighbor_j ,neighbor_k = current_i + di, current_j + dj, current_k + dk

            if not (0 <= neighbor_i < nx and 0 <= neighbor_j < ny and 0 <= neighbor_k < nz):
                continue

            if sdf_function((x[neighbor_i], y[neighbor_j],z[neighbor_k])) > 0:
                continue

            if (neighbor_i, neighbor_j,neighbor_k) in closed_set:
                continue

            cost = move_costs[current_k,current_j, current_i, d_idx]
            if cost == np.inf:
                continue

            tentative_g_score = g_score[current_k,current_j, current_i] + cost

            if tentative_g_score >= g_score[neighbor_k,neighbor_j, neighbor_i]:
                continue

            came_from[(neighbor_i, neighbor_j,neighbor_k)] = (current_i, current_j,current_k)
            g_score[neighbor_k,neighbor_j, neighbor_i] = tentative_g_score
            travel_time[neighbor_k,neighbor_j, neighbor_i] = tentative_g_score

            h = heuristic(neighbor_i, neighbor_j, neighbor_k, i_goal, j_goal,k_goal, dx, dy,dz)
            f_score[neighbor_k,neighbor_j, neighbor_i] = tentative_g_score + heuristic_weight * h

            heapq.heappush(
                open_set, (f_score[neighbor_k,neighbor_j, neighbor_i], neighbor_i, neighbor_j,neighbor_k)
            )

    return [], travel_time


import numpy as np

def precalculate_move_costs(
    v0, vx, vy, vz, dir_offsets, dx, dy, dz, weight_sdf, pow_v0, pow_al
):
    print(
        "Pow v0 :",pow_v0
    )
    print(
        "Pow al :",pow_al
    )
    nz, ny, nx = v0.shape
    n_directions = len(dir_offsets)

    dir_offsets = np.array(dir_offsets)
    delta_xyz = dir_offsets * np.array([dz, dy, dx])  # shape (n_directions, 3)
    distances = np.linalg.norm(delta_xyz, axis=1)
    valid_dirs = distances > 0
    directions = np.zeros_like(delta_xyz)
    directions[valid_dirs] = delta_xyz[valid_dirs] / distances[valid_dirs, None]

    move_costs = np.full((nz, ny, nx, n_directions), np.inf)

    U=1
    if vx is None or vy is None or vz is None:
        for d_idx in range(n_directions):
            if distances[d_idx] > 0:
                move_costs[..., d_idx] = distances[d_idx]
        return move_costs

    v_field = np.stack([vz, vy, vx], axis=-1)  # shape (nz, ny, nx, 3)

    norm_v_field = np.linalg.norm(v_field, axis=-1)  # shape (nz, ny, nx)

    for d_idx in range(n_directions):
        if not valid_dirs[d_idx]:
            continue

        d = directions[d_idx]  # shape (3,)
        dist = distances[d_idx]

        d_broadcast = d.reshape((1, 1, 1, 3))  # shape (1,1,1,3)
        v = v_field + d_broadcast * U  # shape (nz, ny, nx, 3)
        v_dot_d = np.sum(v * d_broadcast, axis=-1)  # shape (nz, ny, nx)
        v_l = v_field
        v_l_dot_d = np.sum(v_l * d_broadcast, axis=-1)
        norm_d = np.linalg.norm(d)

        with np.errstate(divide='ignore', invalid='ignore'):
            alignment = np.abs(v_l_dot_d) / (norm_v_field * norm_d)
            alignment[norm_v_field < 1e-4] = 1.0

        alignment = alignment ** pow_al
        effective_speed = np.maximum(v0 ** pow_v0 * v_dot_d * alignment,0.000000000000001)

        if pow_al > 0 or pow_v0 > 0:
            mask = (v_l_dot_d > 0) & (v0 > 0)
        else:
            mask = v_dot_d > 0

        cost = np.full_like(effective_speed, np.inf)
        cost[mask] = dist / effective_speed[mask]
        move_costs[..., d_idx] = cost

    return move_costs


def heuristic(i1, j1,k1, i2, j2,k2, dx, dy,dz):
    """
    Calcule une heuristique admissible pour A*.
    Utilise la distance euclidienne divisée par la vitesse maximale possible.
    """
    distance = np.sqrt(((i2 - i1) * dx) ** 2 + ((j2 - j1) * dy) ** 2 +  ((k2 - k1) * dz) ** 2)
    v_max = 1.0
    return distance / v_max


def plot_velocity(step, vx, vy, v0, X, Y):
    step = 8

    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    vx_sub = vx[::step, ::step]
    vy_sub = vy[::step, ::step]
    v0_sub = v0[::step, ::step]
    print(v0_sub.shape)
    mask = ((vx_sub != 0) | (vy_sub != 0)) & (v0_sub > 0.2)

    X_masked = X_sub[mask]
    Y_masked = Y_sub[mask]
    vx_masked = vx_sub[mask]
    vy_masked = vy_sub[mask]
    plt.imshow(
        v0,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap="Reds",
        alpha=0.7,
    )
    plt.colorbar(label="v0")

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
    X, Y, sdf_function, path, vx, vy, v0, scale, label="", color=None
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


def compute_v(x, y,z, vx_phys,vy_phys,vz_phys, B, grid_size, ratio, sdf_function, c):

    if len(x) != grid_size[0] and len(y) != grid_size[1]:
        raise ValueError("x,y are not coherent with the size of the grid")
    save_path_phi = f"data/phi/grid_size_{grid_size[0]}_{grid_size[1]}_{grid_size[2]}_phi_3d.npy"
    if os.path.exists(save_path_phi):
        phi = np.load(save_path_phi)
    else:
        os.makedirs(os.path.dirname(save_path_phi), exist_ok=True)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        phi_values = np.array([sdf_function(point) for point in points])
        phi = phi_values.reshape(Z.shape)
        np.save(save_path_phi, phi)
    # print("X :",len(x))
    # print("Y :",len(y))
    # print("Z :",len(z))
    phi=phi.T
    # print("Shape of phi :",phi.shape)
    
    v0 = (1.0 / (1.0 + np.exp(B * phi))) - c
    # v0 = np.clip(v0, -0.001, 1.0)
    save_path_flow = f"data/velocity_flow/grid_size_{grid_size[0]}_{grid_size[1]}_{grid_size[2]}_phi_3d/"
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
        flow_direction_z = np.load(
            os.path.join(save_path_flow, "flow_direction_z.npy")
        )

    else:
        os.makedirs(save_path_flow, exist_ok=True)
        magnitude = np.sqrt(vx_phys**2 + vy_phys**2 + vz_phys**2)
        mask = magnitude > 0

        flow_strength = magnitude
        flow_direction_x = np.zeros_like(vx_phys)
        flow_direction_y = np.zeros_like(vy_phys)
        flow_direction_z = np.zeros_like(vz_phys)

        flow_direction_x[mask] = vx_phys[mask] / magnitude[mask]
        flow_direction_y[mask] = vy_phys[mask] / magnitude[mask]
        flow_direction_z[mask] = vz_phys[mask] / magnitude[mask]

        np.save(os.path.join(save_path_flow, "flow_strength.npy"), flow_strength)
        np.save(os.path.join(save_path_flow, "flow_direction_x.npy"), flow_direction_x)
        np.save(os.path.join(save_path_flow, "flow_direction_y.npy"), flow_direction_y)
        np.save(os.path.join(save_path_flow, "flow_direction_z.npy"), flow_direction_z)

        print("Flow ready")
    vx = flow_direction_x * flow_strength
    vy = flow_direction_y * flow_strength
    vz = flow_direction_z * flow_strength
    vx = vx.T
    vy = vy.T
    vz = vz.T
    
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    quantile_999 = np.quantile(speed, 0.999)
    print(f"Seuil 99,9% (quantile) : {quantile_999:.3e}")

    mask = speed <= quantile_999

    vx = np.where(mask, vx, 0.0)
    vy = np.where(mask, vy, 0.0)
    vz = np.where(mask, vz, 0.0)
    print("vx shape :",vx.shape)
    print("vy shape :",vy.shape)
    print("vz shape :",vz.shape)
        
        
    return v0, vx, vy,vz, save_path_phi, save_path_flow

def paraview_export(vx,vy,vz,dx,dy,dz,sdf,path,output_save_path):
    N =vx.shape
    nx,ny,nz = N
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (dx, dy, dz)
    grid.origin = (0, 0, 0)
    sdf=sdf.T
    grid["SDF"] = sdf.flatten(order="F")
    print(grid["SDF"].shape)
    velocity_vectors = np.stack([vx, vy, vz], axis=-1)
    grid["velocity"] = velocity_vectors.reshape(-1, 3, order="F")
    print(grid["velocity"].shape)  # doit être (nx * ny * nz, 3)
    
    grid["path"] = path.flatten(order="F")
    # Sauvegarde
    grid.save(output_save_path)
    print(f"Fichier sauvegardé dans : {output_save_path}")
    

def plot_different_path(
    files_path, label_list, x_new, y_new, sdf_function, vx, vy, v0, scale
):
    path_list = []
    palette = sns.color_palette()

    for id, file_path in enumerate(files_path):
        path = np.load(file_path)
        path, distances = resample_path(path, len(path))
        print(f"{file_path} length : {distances}")
        visualize_results_a_star(
            x_new,
            y_new,
            sdf_function,
            path,
            vx,
            vy,
            v0,
            scale,
            label=label_list[id],
            color=palette[id],
        )

    plt.legend()
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Path")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_ani_test_{current_time}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Exemple : 16 directions ou plus selon le rayon

    ratio = 5

    # Determine the size of the domain. It maps each point of the domain to each point on the grid.
    sdf_func,sdf_phys,velocity_retina,x_phys,y_phys,z_phys,vx_phys,vy_phys,vz_phys,physical_depth,physical_width,physical_height,scale = load_sim_sdf(ratio)
    print("SDF phys :", sdf_phys.shape) ## Convention x,y,z (576,528,24)
    print("x_phys   :", x_phys.shape) ## 576
    print("y_phys   :", y_phys.shape) ## 528
    print("z_phys   :", z_phys.shape) ## 24
    X, Y,Z= np.meshgrid(x_phys, y_phys,z_phys,indexing='ij')
    print("X shape :",X.shape) # (576, 528, 24) 
    # print("Y shape :",Y.shape) # (576, 528, 24)
    # print("Z shape :",Z.shape) # (576, 528, 24)
    
    def sdf_func_2D(point):
        return sdf_func((point[0],point[1],z_phys[len(z_phys)//2]))
    # sdf_function :  Calculate the sdf in any point of the domain py interpolation
    # velocity_retina :  Calculate the velocity in any point of the domain py interpolation


    plt.figure(figsize=(12, 10))
    weight_sdf = 1
    start_point = (physical_width * 0.8, physical_height * 0.3,physical_depth*0.5)
    goal_point = (0.8,0.6,0.5)
    goal_point = (physical_width * goal_point[0], physical_height * goal_point[1],physical_depth * goal_point[2])
    print(
        "Distance between the two points : ",
        np.linalg.norm(np.array(goal_point) - np.array(start_point)),
    )
    
    print("SDF of goal point : ", sdf_func(goal_point))
    c = 0.5
    B = 10
    h = 1
    pow_v0 = 0
    pow_al = 10
    max_radius = 1.5
    # pow_v0 = 0
    # pow_al = 0

    shortest_geo_path = False
    v1 = False
    
    print('Go compute phi')
    grid_size = (len(x_phys), len(y_phys),len(z_phys))
    v0,vx,vy,vz ,_, _ = compute_v(x_phys, y_phys, z_phys,vx_phys,vy_phys,vz_phys, B, grid_size, ratio, sdf_func, c)

    ## Convention for A* : k,j,i -> z,y,x
    
    # print("Vx shape : ",vx.shape) #(24, 528, 576) (z,y,x)
    # print("Vy shape : ",vy.shape) #(24, 528, 576)
    # print("Vz shape : ",vz.shape) #(24, 528, 576)
    # print("V0 shape : ",v0.shape) #(24, 528, 576)
    
    
    
    if shortest_geo_path:
        v0 = np.ones_like(v0)
        vx = None
        vy = None
        h = 0
        pow_v0 = 0
        pow_al = 0
        max_radius = 5

    if v1:
        pow_v0 = 0
        pow_al = 0
        h = 0
        max_radius = 5

   
    start_time = time.time()


    
    path, travel_time = astar_anisotropic(
        x_phys,
        y_phys,
        z_phys,
        v0,
        vx,
        vy,
        vz,
        start_point,
        goal_point,
        sdf_func,
        sdf_phys,
        heuristic_weight=h,
        pow_v0=pow_v0,
        pow_al=pow_al,
        max_radius=max_radius,
    )

    # path = shortcut_path(path,is_collision_free,sdf_interpolator)
    print("Travel time :", travel_time)

    path = np.array(path)  # de forme (N, 2)
    dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
    print("Path shape : ", path.shape)
    
    print("Path before resampling :", len(path))
    # n = ceil(np.max(dist) / (5 * 1e-3))
    # if n > 1:
    #     path, distances = resample_path(path, len(path) * n)
    # print("After resampling : ", len(path))
    # print("Path length : ", distances)
    z_coords =path[:,2] 
    
    smoothed_x = gaussian_filter1d(path[:, 0], sigma=1)
    smoothed_y = gaussian_filter1d(path[:, 1], sigma=1)
    smoothed_z = gaussian_filter1d(path[:, 2], sigma=1)
    path = np.stack([smoothed_x, smoothed_y], axis=1)
    
    visualize_results_a_star(
        X, Y, sdf_func_2D, path, vx, vy, v0, scale, label=f"B : {B}"
    )
    # plot_velocity(6, vx, vy, v0, X, Y)
    plt.legend()
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Path")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_ani_test_{current_time}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.hist(z_coords, bins=50, density=True)
    plt.savefig('fig/pathz.png',dpi=100,bbox_inches='tight')
    plt.close()
    v0  = v0 ** pow_v0
    plt.hist(v0.flatten(), bins=100)
    plt.yscale("log")
    plt.title("Histogramme des vitesses")
    plt.xlabel("||v0||")
    plt.ylabel("Frequency (log)")
    plt.savefig('fig/hist_v0')
    plt.close()
    # save_path_path = f"data/retina2D_path_time_4_v1_bis_bg.npy"
    # np.save(save_path_path, path, allow_pickle=False)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("Execution time:", elapsed_time, "minutes")
    # files_path = ['data/retina2D_path_time_4_free_bg.npy','data/retina2D_path_time_4_v1_bis_bg.npy','data/retina2D_path_time_4_v2_bg.npy']
    # label_list = ['Shortest geometrical path','Algorithm 1', 'Algorithm 2']
    # plot_different_path(files_path,label_list,x_phys,y_phys,sdf_func,vx,vy,v0,scale)

    save_output_path_para = f"fig/Astar_path_{current_time}.vti"
    dx = x_phys[1] - x_phys[0]
    dy = y_phys[1] - y_phys[0]
    dz = z_phys[1] - z_phys[0]
    # paraview_export(vx,vy,vz,dx,dy,dz,sdf_phys,path,save_output_path_para)