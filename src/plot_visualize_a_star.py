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


def plot_contour_path(
    X, Y, sdf_function, path, scale, label="", color=None
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


def plot_a_star( X, Y, sdf_function, vx, vy, v0, path, scale, current_time,label="", color=None,bool_velocity=False,plot_path=False):
    if plot_path:
        plot_contour_path(
            X, Y, sdf_function, path, scale, label=label
        )
    if bool_velocity:
        plot_velocity(6, vx, vy, v0, X, Y)
        
    plt.legend()
    
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title("Path")
    plt.tight_layout()
    plt.savefig(f"fig/Astar_ani_test_{current_time}.png", dpi=300, bbox_inches="tight")
    plt.close()

    

# Fonction alternative si vous voulez définir les données sur les points
def paraview_export_points(vx, vy, vz, dx, dy, dz, sdf, path_points, output_save_path):
    """
    Version alternative avec données définies sur les points (pas les cellules)
    """
    nz, ny, nx = vx.shape
    
    # Pour des données sur les points, dimensions = nombre de points
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (dx, dy, dz)
    grid.origin = (0, 0, 0)
    
    # Transposer et assigner aux points
    sdf_vtk = sdf.transpose(2, 1, 0).flatten()
    vx_vtk = vx.transpose(2, 1, 0).flatten()
    vy_vtk = vy.transpose(2, 1, 0).flatten()
    vz_vtk = vz.transpose(2, 1, 0).flatten()
    
    grid.point_data["SDF"] = sdf_vtk
    velocity = np.column_stack((vx_vtk, vy_vtk, vz_vtk))
    grid.point_data["Velocity"] = velocity
    
    grid.save(output_save_path)
    
    return grid
    

def paraview_export(vx, vy, vz, dx, dy, dz, sdf, path_points, output_save_path):
    """
    Exporte les données pour ParaView avec format VTI correct.
    
    IMPORTANT: PyVista ImageData utilise une convention différente de NumPy
    - NumPy: indexation [z, y, x] pour tableaux (nz, ny, nx)
    - VTK/ParaView: ordre (x, y, z) pour les dimensions et données
    """
    
    nz, ny, nx = vx.shape
    print(f"Dimensions NumPy originales: nz={nz}, ny={ny}, nx={nx}")
    print(f"Shape SDF: {sdf.shape}")
    
    # DIAGNOSTIC: Vérifier les dimensions des données SDF
    if sdf.size != nx * ny * nz:
        print(f"ATTENTION: Taille SDF ({sdf.size}) != vx.size ({nx * ny * nz})")
        print(f"SDF semble avoir des dimensions différentes")
        
        # Option 1: Si SDF a une dimension de plus (points vs cellules)
        if sdf.shape == (nz+1, ny+1, nx+1):
            print("SDF défini sur les points - conversion vers cellules")
            # Moyenne des 8 coins de chaque cellule pour obtenir la valeur au centre
            sdf_cells = 0.125 * (
                sdf[:-1, :-1, :-1] + sdf[1:, :-1, :-1] + 
                sdf[:-1, 1:, :-1] + sdf[1:, 1:, :-1] +
                sdf[:-1, :-1, 1:] + sdf[1:, :-1, 1:] + 
                sdf[:-1, 1:, 1:] + sdf[1:, 1:, 1:]
            )
            sdf = sdf_cells
            print(f"Nouvelle shape SDF: {sdf.shape}")
        
        # Option 2: Si SDF a les bonnes dimensions mais pas le bon ordre
        elif sdf.size == nx * ny * nz:
            sdf = sdf.reshape(nz, ny, nx)
            print("SDF reshapé aux bonnes dimensions")
        
        # Option 3: Tronquer si trop grand
        else:
            print("Troncature du SDF aux dimensions des vitesses")
            sdf = sdf[:nz, :ny, :nx] if sdf.ndim == 3 else sdf.reshape(nz, ny, nx)
    
    # Créer la grille VTK
    grid = pv.ImageData()
    
    # Pour les données définies sur les cellules, les dimensions sont le nombre de cellules
    grid.dimensions = (nx, ny, nz)  # Nombre de cellules dans chaque direction
    grid.spacing = (dx, dy, dz)
    grid.origin = (0, 0, 0)
    
    print(f"Grille VTK - Dimensions: {grid.dimensions}")
    print(f"Nombre de cellules: {grid.n_cells}")
    print(f"Nombre de points: {grid.n_points}")
    
    # Vérification finale des dimensions
    expected_cells = (nx-1) * (ny-1) * (nz-1) if grid.n_cells != nx * ny * nz else nx * ny * nz
    
    # Réorganiser les données selon l'ordre VTK (x varie le plus vite)
    # VTK attend l'ordre: pour chaque z, pour chaque y, pour chaque x
    # NumPy: (nz, ny, nx) → VTK: (nx, ny, nz) puis flatten
    
    # Transposer de (z,y,x) vers (x,y,z) puis flatten
    sdf_vtk = sdf.transpose(2, 1, 0).flatten()
    vx_vtk = vx.transpose(2, 1, 0).flatten()
    vy_vtk = vy.transpose(2, 1, 0).flatten()
    vz_vtk = vz.transpose(2, 1, 0).flatten()
    
    print(f"Vérification dimensions:")
    print(f"  SDF VTK shape: {sdf_vtk.shape}, attendu: {grid.n_cells}")
    print(f"  Vitesses VTK shape: {vx_vtk.shape}")
    
    # Assigner les données aux cellules
    if sdf_vtk.size == grid.n_cells:
        grid.cell_data["SDF"] = sdf_vtk
        
        # Créer le vecteur vitesse
        velocity = np.column_stack((vx_vtk, vy_vtk, vz_vtk))
        grid.cell_data["Velocity"] = velocity
        
        print("✓ Données assignées avec succès")
    else:
        raise ValueError(f"Incompatibilité de taille: SDF={sdf_vtk.size}, cellules={grid.n_cells}")
    
    # Ajouter le chemin si fourni
    if path_points is not None and len(path_points) > 0:
        # Convertir le chemin en format VTK
        path_points_array = np.array(path_points)
        if path_points_array.shape[1] == 3:  # Points 3D
            path_polydata = pv.PolyData(path_points_array)
            # Créer des lignes connectant les points
            lines = np.array([[2, i, i+1] for i in range(len(path_points)-1)])
            path_polydata.lines = lines.flatten()
            
            # Sauvegarder le chemin séparément
            path_output = output_save_path.replace('.vti', '_path.vtp')
            path_polydata.save(path_output)
            print(f"Chemin sauvegardé: {path_output}")
    
    # Sauvegarder la grille
    grid.save(output_save_path)
    print(f"Grille sauvegardée: {output_save_path}")
    
    return grid
    
    
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