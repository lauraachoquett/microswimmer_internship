import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


def fmm_path_indices(start, goal, sdf, min_distance=0.1):
    """
    Utilise la méthode Fast Marching Method (FMM) pour trouver un chemin entre start et goal
    en suivant le gradient de la carte de temps calculée par FMM.

    Args:
        start: Indices [x, y] du point de départ dans la grille
        goal: Indices [x, y] du point d'arrivée dans la grille
        sdf: Carte de distance signée 2D
        min_distance: Distance minimale à maintenir par rapport aux obstacles

    Returns:
        Une liste des indices [x, y] formant le chemin
    """
    # Dimensions de la grille
    ny, nx = sdf.shape

    # Créer une carte de vitesse basée sur la SDF
    # Plus on est proche d'un obstacle, plus on va lentement
    speed = np.ones_like(sdf)
    obstacles = sdf < min_distance
    speed[obstacles] = 0.0001  # Vitesse très faible près des obstacles

    # Zones loin des obstacles ont une vitesse plus élevée
    # On normalise pour avoir des vitesses entre 0.0001 et 1
    safe_regions = ~obstacles
    if np.any(safe_regions):
        normalized_sdf = np.clip(sdf, min_distance, None)
        max_sdf = np.max(normalized_sdf[safe_regions])
        if max_sdf > min_distance:
            # Vitesse croissante avec la distance aux obstacles
            speed[safe_regions] = 0.2 + 0.8 * (
                normalized_sdf[safe_regions] - min_distance
            ) / (max_sdf - min_distance)

    # Création de la carte de temps par Fast Marching Method
    # On utilise distance_transform_edt comme approximation de la FMM
    # En partant du but et en propageant vers le départ
    goal_map = np.zeros_like(speed, dtype=bool)
    goal_map[goal[1], goal[0]] = True

    # Calculer la carte de temps (plus grande valeur = plus loin du but)
    time_map = distance_transform_edt(~goal_map, sampling=[1 / s for s in speed])

    # Tracer le chemin en remontant le gradient négatif de la carte de temps
    # On commence au point de départ
    path = []
    current = list(start)  # Copie de start pour éviter de le modifier
    path.append(current.copy())

    # Taille des pas pour le suivi de gradient (plus petit = plus précis)
    step_size = 1.0
    max_iterations = 10000  # Limite pour éviter les boucles infinies

    # Gradient de la carte de temps aux points x et y
    y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    # Fonction pour calculer le gradient à un point donné
    def compute_gradient(point):
        x, y = point
        # Assurez-vous que les indices sont dans les limites
        if x < 1 or x >= nx - 1 or y < 1 or y >= ny - 1:
            return np.array([0, 0])

        # Calculer le gradient par différences finies centrées
        dx = (time_map[y, x + 1] - time_map[y, x - 1]) / 2.0
        dy = (time_map[y + 1, x] - time_map[y - 1, x]) / 2.0

        # Retourner le gradient négatif (direction de descente)
        return -np.array([dx, dy])

    i = 0
    while i < max_iterations:
        # Calculer le gradient au point courant
        gradient = compute_gradient([current[0], current[1]])

        # Normaliser le gradient
        norm = np.linalg.norm(gradient)
        if norm < 1e-10:  # Si le gradient est presque nul
            break

        gradient = gradient / norm

        # Mettre à jour la position courante
        next_x = current[0] + step_size * gradient[0]
        next_y = current[1] + step_size * gradient[1]

        # Arrondir pour obtenir les indices de la grille
        next_idx = [int(round(next_x)), int(round(next_y))]

        # Vérifier si nous sommes arrivés au but
        if abs(next_idx[0] - goal[0]) <= 1 and abs(next_idx[1] - goal[1]) <= 1:
            path.append(goal)
            break

        # Vérifier si nous sommes toujours dans la grille
        if next_idx[0] < 0 or next_idx[0] >= nx or next_idx[1] < 0 or next_idx[1] >= ny:
            break

        # Vérifier si nous sommes trop près d'un obstacle
        if sdf[next_idx[1], next_idx[0]] < min_distance:
            # Trouver une direction alternative
            break

        # Ajouter le point au chemin et continuer
        current = next_idx
        path.append(current.copy())

        # Si on ne bouge plus, on arrête
        if len(path) > 1 and path[-1] == path[-2]:
            break

        i += 1

    # Simplifier le chemin pour retirer les points redondants
    simplified_path = [path[0]]
    for i in range(1, len(path)):
        if path[i] != simplified_path[-1]:
            simplified_path.append(path[i])

    # Vérifier si le chemin a été trouvé
    if len(simplified_path) <= 1 or not (
        abs(simplified_path[-1][0] - goal[0]) <= 1
        and abs(simplified_path[-1][1] - goal[1]) <= 1
    ):
        print("Attention: Chemin incomplet, l'algorithme n'a pas atteint le but")

    return simplified_path
