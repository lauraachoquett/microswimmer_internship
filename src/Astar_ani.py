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

from src.utils import gcd_of_three,generate_directions_3d
from src.Astar import resample_path
from src.data_loader import load_sdf_from_csv, load_sim_sdf, vel_read,plot_sdf_slices
from src.fmm import sdf_func_and_velocity_func
from src.plot_visualize_a_star import plot_a_star,paraview_export, paraview_export_points

def heuristic(i1, j1,k1, i2, j2,k2, dx, dy,dz):
    """
    Calcule une heuristique admissible pour A*.
    Utilise la distance euclidienne divisée par la vitesse maximale possible.
    """
    distance = np.sqrt(((i2 - i1) * dx) ** 2 + ((j2 - j1) * dy) ** 2 +  ((k2 - k1) * dz) ** 2)
    v_max = 1.0
    return distance / v_max

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
            path_id = [[i_goal,j_goal,k_goal]]
            i, j,k = i_goal, j_goal,k_goal
            while (i, j,k) in came_from:
                i, j,k = came_from[(i, j,k)]
                p = (x[i], y[j],z[k])
                path.append(p)
                path_id.append([i,j,k])
                # print("V0 : ", v0[k,j,i]**pow_v0)
            path_id.reverse()
            path.reverse()
            return path, path_id,travel_time[k_goal,j_goal, i_goal]

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

    return [],[],travel_time



def precalculate_move_costs(v0, vx, vy, vz, dir_offsets, dx, dy, dz, weight_sdf, pow_v0, pow_al):
    """
    Pré-calcule les coûts de déplacement pour chaque direction à chaque point en 3D.
    Version vectorisée optimisée.

    Args:
        v0: tableau 3D (nz, ny, nx) - vitesse de base
        vx, vy, vz: tableaux 3D (nz, ny, nx) - composantes du champ de vitesse
        dir_offsets: liste de tuples (di, dj, dk) - directions de déplacement
        dx, dy, dz: pas spatiaux
        weight_sdf: poids SDF (non utilisé dans cette version)
        pow_v0, pow_al: exposants pour v0 et alignement

    Retourne:
        move_costs: tableau 4D (nz, ny, nx, n_directions) des coûts
    """
    nz, ny, nx = v0.shape
    n_directions = len(dir_offsets)
    
    # Convertir dir_offsets en array pour vectorisation
    dir_offsets = np.array(dir_offsets)  # shape: (n_directions, 3)
    
    # Calculer toutes les distances d'un coup
    distances = np.sqrt(
        (dir_offsets[:, 0] * dx) ** 2 + 
        (dir_offsets[:, 1] * dy) ** 2 + 
        (dir_offsets[:, 2] * dz) ** 2
    )
    
    # Masque pour les directions valides (distance > 0)
    valid_dirs = distances > 0
    
    # Initialiser le tableau de coûts
    move_costs = np.full((nz, ny, nx, n_directions), np.inf)
    
    # Cas simple : pas de champ de vitesse
    if vx is None or vy is None or vz is None:
        move_costs[:, :, :, valid_dirs] = distances[valid_dirs]
        return move_costs
    
    U = 1
    
    # Calculer les directions normalisées pour toutes les directions valides
    dir_norm = np.zeros((n_directions, 3))
    dir_norm[valid_dirs, 0] = dir_offsets[valid_dirs, 0] * dx / distances[valid_dirs]
    dir_norm[valid_dirs, 1] = dir_offsets[valid_dirs, 1] * dy / distances[valid_dirs]
    dir_norm[valid_dirs, 2] = dir_offsets[valid_dirs, 2] * dz / distances[valid_dirs]
    
    # Vectoriser sur toutes les positions et directions
    for d_idx in np.where(valid_dirs)[0]:
        di, dj, dk = dir_offsets[d_idx]
        distance = distances[d_idx]
        dir_x, dir_y, dir_z = dir_norm[d_idx]
        
        # Calculer les indices des voisins (avec clamp)
        neighbor_i = np.clip(np.arange(nx)[None, None, :] + di, 0, nx-1)
        neighbor_j = np.clip(np.arange(ny)[None, :, None] + dj, 0, ny-1)
        neighbor_k = np.clip(np.arange(nz)[:, None, None] + dk, 0, nz-1)
        
        # Champ de vitesse local en chaque point
        v_l_x = vx  # shape: (nz, ny, nx)
        v_l_y = vy  # shape: (nz, ny, nx)
        v_l_z = vz  # shape: (nz, ny, nx)
        
        # Vitesse totale v = v_l + U * d
        v_x = v_l_x + U * dir_x
        v_y = v_l_y + U * dir_y
        v_z = v_l_z + U * dir_z
        
        # Composante du flux dans la direction
        flow_component = v_x * dir_x + v_y * dir_y + v_z * dir_z
        
        # Calcul de l'alignement
        v_l_norm = np.sqrt(v_l_x**2 + v_l_y**2 + v_l_z**2)
        v_l_dot_d = v_l_x * dir_x + v_l_y * dir_y + v_l_z * dir_z
        
        # Éviter division par zéro
        alignment = np.ones_like(v_l_norm)
        mask_nonzero = v_l_norm > 0.0001
        alignment[mask_nonzero] = (np.abs(v_l_dot_d[mask_nonzero]) / v_l_norm[mask_nonzero]) ** pow_al
        
        # Vitesse effective
        effective_speed = flow_component * alignment
        
        # Facteur de vitesse du voisin
        v0_neighbor = v0[neighbor_k, neighbor_j, neighbor_i]
        effective_speed = (v0_neighbor ** pow_v0) * np.maximum(effective_speed, 0.001)
        
        # Conditions pour calculer le coût
        if pow_al > 0 or pow_v0 > 0:
            valid_mask = (v_l_dot_d > 0.0) & (v0_neighbor > 0)
        else:
            valid_mask = flow_component > 0
        
        # Calculer les coûts
        costs = np.full((nz, ny, nx), np.inf)
        if pow_al > 0 or pow_v0 > 0:
            costs[valid_mask] = distance / effective_speed[valid_mask]
        else:
            costs[valid_mask] = distance / flow_component[valid_mask]
        
        move_costs[:, :, :, d_idx] = costs
    
    return move_costs


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
    
    v0 = - phi
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
    vx = flow_direction_x * flow_strength / (np.max(flow_strength)) * ratio
    vy = flow_direction_y * flow_strength / (np.max(flow_strength)) * ratio
    vz = flow_direction_z * flow_strength / (np.max(flow_strength)) * ratio
    vx = vx.T
    vy = vy.T
    vz = vz.T
    # === Statistiques initiales ===
    for name, arr in zip(["vx", "vy", "vz"], [vx, vy, vz]):
        print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}, std={arr.std():.3e}")

    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    print(f"Speed: min={speed.min():.3e}, max={speed.max():.3e}, mean={speed.mean():.3e}, std={speed.std():.3e}")
    
    print("vx shape :",vx.shape)
    print("vy shape :",vy.shape)
    print("vz shape :",vz.shape)
        
        
    return v0, vx, vy,vz, save_path_phi, save_path_flow


def resample_and_smooth(path,sigma): 
    dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
    print("Path shape : ", path.shape)
    
    print("Path before resampling :", len(path))
    n = ceil(np.max(dist) / (5 * 1e-3))
    if n > 1:
        path, distances = resample_path(path, len(path) * n)
    print("After resampling : ", len(path))
    print("Path length : ", distances)
    z_coords =path[:,2] 
    smoothed_x = gaussian_filter1d(path[:, 0], sigma=sigma)
    smoothed_y = gaussian_filter1d(path[:, 1], sigma=sigma)
    smoothed_z = gaussian_filter1d(path[:, 2], sigma=sigma)
    path_2D = np.stack([smoothed_x, smoothed_y], axis=1)
    path = np.stack([smoothed_x, smoothed_y,smoothed_z], axis=1)
    return path,path_2D,z_coords

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
    

    plt.figure(figsize=(12, 10))
    weight_sdf = 1
    start_point = (physical_width * 0.98, physical_height * 0.3,physical_depth*0.5)
    goal_point = (0.8,0.3,0.5)
    goal_point = (physical_width * goal_point[0], physical_height * goal_point[1],physical_depth * goal_point[2])
    print(
        "Distance between the two points : ",
        np.linalg.norm(np.array(goal_point) - np.array(start_point)),
    )
    
    print("SDF of goal point : ", sdf_func(goal_point))
    c = 0.5
    B = 5
    h = 1
    pow_v0 = 7
    pow_al = 4
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
    
    path, path_id,travel_time = astar_anisotropic(
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
    path,path_2D,z_coords = resample_and_smooth(path,sigma=5)
    
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    plot_a_star( X, Y, sdf_func_2D, vx, vy, v0, path_2D, scale, current_time,label=f"pow al : {pow_al}, pow_v0 : {pow_v0}, B : {B} and c :{c}", bool_velocity=False,plot_path=True)
    
    # save_path_path = f"data/retina2D_path_time_4_v1_bis_bg.npy"
    # np.save(save_path_path, path, allow_pickle=False)


    # files_path = ['data/retina2D_path_time_4_free_bg.npy','data/retina2D_path_time_4_v1_bis_bg.npy','data/retina2D_path_time_4_v2_bg.npy']
    # label_list = ['Shortest geometrical path','Algorithm 1', 'Algorithm 2']
    # plot_different_path(files_path,label_list,x_phys,y_phys,sdf_func,vx,vy,v0,scale)


    save_output_path_para_file= f"paraview/Astar_path_{current_time}"
    os.makedirs(save_output_path_para_file,exist_ok=True)
    save_output_path_para = os.path.join(save_output_path_para_file,'sdf_vel_path.vti')
    dx = x_phys[1] - x_phys[0]
    dy = y_phys[1] - y_phys[0]
    dz = z_phys[1] - z_phys[0]
    grid =  paraview_export_points(vx,vy,vz,dx,dy,dz,sdf_phys.T,path_id,save_output_path_para)
        
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
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("Execution time:", elapsed_time, "minutes")