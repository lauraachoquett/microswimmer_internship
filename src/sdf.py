import math

import matplotlib.pyplot as plt
import numpy as np

from .Astar import astar, resample_path, shortcut_path,plot_valid_invalid_points
from .fmm import fmm_path_indices



    
def sdf_circle(point, center, radius):
    px, py = point
    cx, cy = center
    distance_to_center = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return distance_to_center - radius


def sdf_many_circle(point, centers, radius):
    distances = np.zeros(len(centers))
    for i, center in enumerate(centers):
        distances[i] = np.linalg.norm(np.array(center) - np.array(point))

    id = np.argmin(distances)
    center = centers[id]
    return sdf_circle(point,center,radius)

def get_contour_coordinates(X, Y, Z, level=0):
    contours = plt.contour(X, Y, Z, levels=[level])
    coordinates = []
    for collection in contours.collections:
        for path in collection.get_paths():
            coordinates.append(path.vertices)
    return  np.squeeze(np.array(coordinates))


def plot_sdf_path(X,Y,Z,path,path_indice_shortcut):
    plt.figure(figsize=(6, 6))


    plot_sdf(X,Y,Z)
    if path is not None:
        plt.plot(X[path[:, 0], path[:, 1]], Y[path[:, 0], path[:, 1]], color='black')
        plt.plot(X[path_indice_shortcut[:, 0], path_indice_shortcut[:, 1]], Y[path_indice_shortcut[:, 0], path_indice_shortcut[:, 1]], color='green')
    else:
        print("No path found!")

    plt.title("SDF of a Circle with A* Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.legend()
    plt.savefig('fig/sdf_circle_path_plot.png', dpi=100, bbox_inches='tight')


def plot_sdf(X, Y, Z):
    contour = plt.contourf(X, Y, Z, levels=100, cmap='coolwarm')
    plt.colorbar(contour, label='Signed Distance')
    plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=1)



if __name__ == '__main__':
    type='circle'
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y,indexing='ij')
    print(X[0,250])
    print(Y[0,250])
    if type =='circle':
        centers = [(-1, 1/2),(1, -1/2)]
        radius = 3/4
        min_distance = 0.1
        Z = np.vectorize(lambda px, py: sdf_many_circle((px, py), centers, radius))(X, Y)
    start = [0, 250]
    goal = [490, 250]
    start_coords = (X[start[0], start[1]], Y[start[0], start[1]])
    goal_coords = (X[goal[0], goal[1]], Y[goal[0], goal[1]])
    print("Start coordinates:", start_coords)
    print("Goal coordinates:", goal_coords)
    path_indices = np.array(astar(start, goal, Z, min_distance))
    
    path_indice_shortcut = shortcut_path(path_indices,X,Y,Z, min_distance)
    path_coords_short = np.array([[X[idx[1], idx[0]], Y[idx[1], idx[0]]] for idx in path_indice_shortcut])

    plot_sdf_path(X,Y,Z,path_indices,path_indice_shortcut)
