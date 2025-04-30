import heapq
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator, splev, splprep
from scipy.ndimage import gaussian_filter1d

from .data_loader import load_sdf_from_csv, vel_read
from .fmm import sdf_func_and_velocity_func


def astar_anisotropic(x, y, v0, vx, vy, start_point, goal_point, sdf_function,heuristic_weight=1.0, directions=16):
    """
    Implémentation de l'algorithme A* adapté pour les écoulements anisotropes.
    
    Paramètres:
    - x, y : tableaux 1D des coordonnées de la grille
    - v0 : vitesse propre du milieu, tableau 2D de taille (len(y), len(x))
    - vx, vy : composantes du champ de fluide, tableaux 2D de taille (len(y), len(x))
    - start_point : tuple (x_start, y_start) représentant le point de départ
    - goal_point : tuple (x_goal, y_goal) représentant le point d'arrivée
    - heuristic_weight : poids de l'heuristique (>=1.0, où 1.0 donne un A* optimal)
    - directions : nombre de directions possibles (8, 16, 32...)
    
    Retourne:
    - path : liste de points (x, y) représentant le chemin optimal
    - travel_time : tableau 2D des temps de trajet pour chaque point visité
    """
    nx, ny = len(x), len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Trouver les indices des points de départ et d'arrivée
    i_start = np.argmin(np.abs(x - start_point[0]))
    j_start = np.argmin(np.abs(y - start_point[1]))
    i_goal = np.argmin(np.abs(x - goal_point[0]))
    j_goal = np.argmin(np.abs(y - goal_point[1]))
    
    # Générer les directions en fonction du paramètre
    if directions == 8:
        dir_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), 
                       (-1, 0), (-1, -1), (0, -1), (1, -1)]
    else:
        # Générer directions uniformément réparties sur le cercle
        angles = np.linspace(0, 2*np.pi, directions, endpoint=False)
        dir_offsets = [(round(np.cos(angle)), round(np.sin(angle))) for angle in angles]
        # Éliminer les doublons et le (0,0)
        dir_offsets = list(set(dir_offsets))
        dir_offsets = [d for d in dir_offsets if d != (0, 0)]
    
    # Calculer les coûts de déplacement pour chaque direction à chaque point
    move_costs = precalculate_move_costs(x, y, v0, vx, vy, dir_offsets, dx, dy)
    
    # Initialisation de l'algorithme A*
    closed_set = set()  # Ensemble des points déjà traités
    open_set = [(0, i_start, j_start)]  # File de priorité (coût estimé, i, j)
    heapq.heapify(open_set)
    
    came_from = {}  # Pour reconstruire le chemin
    g_score = np.full((ny, nx), np.inf)  # Coût réel depuis le départ
    g_score[j_start, i_start] = 0
    
    f_score = np.full((ny, nx), np.inf)  # Coût estimé total (g + h)
    h_goal = heuristic(i_start, j_start, i_goal, j_goal, dx, dy)
    f_score[j_start, i_start] = h_goal
    
    # Pour visualisation du champ de temps
    travel_time = np.full((ny, nx), np.inf)
    travel_time[j_start, i_start] = 0
    
    # Boucle principale de l'algorithme A*
    while open_set:
        # Récupérer le nœud avec le plus petit f_score
        _, current_i, current_j = heapq.heappop(open_set)
        
        # Si on a atteint l'objectif
        if current_i == i_goal and current_j == j_goal:
            # Reconstruire le chemin
            path = [(x[i_goal], y[j_goal])]
            i, j = i_goal, j_goal
            while (i, j) in came_from:
                i, j = came_from[(i, j)]
                path.append((x[i], y[j]))
            path.reverse()
            return path, travel_time
        
        # Marquer le nœud comme exploré
        closed_set.add((current_i, current_j))
        
        # Explorer les voisins
        for di, dj in dir_offsets:
            neighbor_i, neighbor_j = current_i + di, current_j + dj
            
            # Vérifier si le voisin est dans la grille
            if not (0 <= neighbor_i < nx and 0 <= neighbor_j < ny) :
                continue
            if sdf_function((x[neighbor_i],y[neighbor_j]))>0:
                continue

            # Si le voisin est déjà exploré
            if (neighbor_i, neighbor_j) in closed_set:
                continue
            
            # Calculer le nouveau g_score
            cost = move_costs[current_j, current_i, dir_offsets.index((di, dj))]
            if cost == np.inf:  # Si déplacement impossible (vitesse négative)
                continue
                
            tentative_g_score = g_score[current_j, current_i] + cost
            
            # Si ce chemin vers le voisin n'est pas meilleur, ignorer
            if tentative_g_score >= g_score[neighbor_j, neighbor_i]:
                continue
                
            # Ce chemin est le meilleur jusqu'à présent, l'enregistrer
            came_from[(neighbor_i, neighbor_j)] = (current_i, current_j)
            g_score[neighbor_j, neighbor_i] = tentative_g_score
            travel_time[neighbor_j, neighbor_i] = tentative_g_score
            
            # Calculer l'heuristique et mettre à jour f_score
            h = heuristic(neighbor_i, neighbor_j, i_goal, j_goal, dx, dy)
            f_score[neighbor_j, neighbor_i] = tentative_g_score + heuristic_weight * h
            
            # Ajouter à la file d'attente
            heapq.heappush(open_set, (f_score[neighbor_j, neighbor_i], neighbor_i, neighbor_j))
    
    # Si on sort de la boucle sans atteindre l'objectif, aucun chemin n'a été trouvé
    return [], travel_time

def precalculate_move_costs(x, y, v0, vx, vy, dir_offsets, dx, dy):
    """
    Pré-calcule les coûts de déplacement pour chaque direction à chaque point.
    
    Retourne:
    - move_costs: tableau 3D (ny, nx, n_directions) des coûts
    """
    ny, nx = v0.shape
    n_directions = len(dir_offsets)
    move_costs = np.full((ny, nx, n_directions), np.inf)
    
    for j in range(ny):
        for i in range(nx):
            for d_idx, (di, dj) in enumerate(dir_offsets):
                # Calculer la distance
                distance = np.sqrt((di*dx)**2 + (dj*dy)**2)
                
                # Normaliser la direction
                if distance > 0:
                    dir_x = di*dx / distance
                    dir_y = dj*dy / distance
                    
                    # Calculer la vitesse effective (vitesse propre + composante du flux)
                    flow_component = vx[j, i] * dir_x + vy[j, i] * dir_y
                    effective_speed = v0[j, i] + flow_component
                    
                    # Si la vitesse effective est positive, calculer le coût
                    if effective_speed > 0:
                        move_costs[j, i, d_idx] = distance / effective_speed
    
    return move_costs

def heuristic(i1, j1, i2, j2, dx, dy):
    """
    Calcule une heuristique admissible pour A*.
    Utilise la distance euclidienne divisée par la vitesse maximale possible.
    """
    # Distance euclidienne en unités de grille
    distance = np.sqrt(((i2 - i1) * dx)**2 + ((j2 - j1) * dy)**2)
    
    # Ici, on suppose une vitesse maximale constante pour l'heuristique
    # En pratique, on pourrait utiliser une estimation plus précise
    v_max = 1.0  # À adapter selon les valeurs de v0, vx, vy
    
    return 0.1*distance / v_max



def visualize_results_a_star(x, y,sdf_function, path,vx,vy):
    """
    Visualise les résultats de l'algorithme A*.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    phi = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            a = sdf_function((x[i], y[j]))
            phi[j, i] = a
    phi = phi / np.max(np.abs(phi))
    
    plt.figure(figsize=(12, 10))
    X, Y = np.meshgrid(x, y)
    # Afficher le champ de temps avec une échelle logarithmique

    contour_sdf = plt.contourf(X, Y, phi, levels=100, cmap="viridis")
    plt.colorbar(contour_sdf, label="Signed Distance")
    plt.contour(X, Y, phi, levels=[0], colors="k", linewidths=2)


    
    # Tracer le chemin
    if len(path)>0:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'r-', linewidth=2)
    
        plt.scatter([path[0][0]], [path[0][1]], color='green', s=100, label='Départ')
        plt.scatter([path[-1][0]], [path[-1][1]], color='red', s=100, label='Arrivée')
    step = 5

    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    vx_sub = vx[::step, ::step]
    vy_sub = vy[::step, ::step]

    mask = (vx_sub != 0) | (vy_sub != 0)

    X_masked = X_sub[mask]
    Y_masked = Y_sub[mask]
    vx_masked = vx_sub[mask]
    vy_masked = vy_sub[mask]

    # Affichage
    plt.quiver(
        X_masked,
        Y_masked,
        vx_masked,
        vy_masked,
        color="white",
        scale=300,
        alpha=0.7,
    )
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("SDF with path")
    plt.tight_layout()
    plt.savefig('fig/Astar_ani.png',dpi=300, bbox_inches='tight')

def compute_v(x,y,velocity_retina,B,grid_size,ratio):

    flow_field=velocity_retina
    save_path_phi = f"data/phi/grid_size_{grid_size[0]}_{grid_size[1]}_phi.npy"
    if os.path.exists(save_path_phi):
        phi = np.load(save_path_phi)
        print('Phi loaded')
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
            print('Flow loaded')

        else:
            flow_strength = np.zeros((len(y), len(x)))
            flow_direction_x = np.zeros((len(y), len(x)))
            flow_direction_y = np.zeros((len(y), len(x)))

            for i in range(len(x)):
                for j in range(len(y)):
                    vx, vy = flow_field((x[i], y[j]))/ratio
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
        
    return speed,vx,vy,save_path_phi,save_path_flow

def shortcut_path(path, is_collision_free,sdf):
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i:
            if is_collision_free(path[i], path[j],sdf):
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
    scale = 8
    ratio = 4
    start_point = (0.98, 0.3)
    goal_point = (0.33, 0.5)
    domain_size = (1 * scale, 1 * scale)
    x, y, N, h, sdf = load_sdf_from_csv(domain_size)
    res_factor=1
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
    B=15
    v0,vx,vy,_,_ = compute_v(x,y,velocity_retina,B,grid_size,ratio)
    v0 = np.ones_like(v0)*1/4
    path, travel_time = astar_anisotropic(x_new, y_new, v0, vx, vy, start_point, goal_point, sdf_function,
                                      heuristic_weight=1.3, directions=16)
    # path = shortcut_path(path,is_collision_free,sdf_interpolator)

    path = np.array(path)  # de forme (N, 2)
    smoothed_x = gaussian_filter1d(path[:, 0], sigma=3)
    smoothed_y = gaussian_filter1d(path[:, 1], sigma=3)
    smoothed_path = np.stack([smoothed_x, smoothed_y], axis=1)
    print("smoooooth")
    # x = path[:, 0]
    # y = path[:, 1]
    # tck, _ = splprep([x, y], s=5)  # s=5 contrôle le lissage
    # u_fine = np.linspace(0, 1, 100)
    # x_smooth, y_smooth = splev(u_fine, tck)
    # smoothed_path = np.vstack([x_smooth, y_smooth]).T
    
    visualize_results_a_star(x_new, y_new, sdf_function,smoothed_path,vx,vy)
    