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

    
def save_grid_paraview(vx, vy, vz, dx, dy, dz, sdf):
    output_save_path_sdf_vel = 'paraview/sdf_vel/sdf_vel.vti'
    import os
    output_dir = os.path.dirname(output_save_path_sdf_vel)
    if not os.path.exists(output_dir): 
        N=vx.shape
        nx,ny,nz = N
        
        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)
        grid.spacing = (dx, dy, dz)
        grid.origin = (0, 0, 0)

        grid["SDF"] = sdf.flatten(order="F")
        velocity_vectors = np.stack([vx, vy, vz], axis=-1)
        print(f"Velocity vectors shape: {velocity_vectors.shape}")
        grid["velocity"] = velocity_vectors.reshape(-1, 3, order="F")
        print(f"Grid velocity shape: {grid['velocity'].shape}")
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Répertoire créé: {output_dir}")
        
        if output_dir and not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Pas de permission d'écriture dans: {output_dir}")
        
        print(f"Tentative de sauvegarde: {output_save_path_sdf_vel}")
        grid.save(output_save_path_sdf_vel)
        print(f"✓ Fichier sauvegardé avec succès: {output_save_path_sdf_vel}")
        

    
    
    

    
    
    
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