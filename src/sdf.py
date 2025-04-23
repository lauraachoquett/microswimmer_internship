import numpy as np
import math
import matplotlib.pyplot as plt
from .Astar import astar

def sdf_circle(point, center, radius):
    px, py = point
    cx, cy = center
    distance_to_center = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return distance_to_center - radius


def plot_sdf_path(X,Y,Z,path):
    plt.figure(figsize=(6, 6))

    contour = plt.contourf(X, Y, Z, levels=100, cmap='coolwarm')
    plt.colorbar(contour, label='Signed Distance')
    plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=1)

    if path is not None:
        plt.plot(x[path[:, 0]], y[path[:, 1]], color='black', label="Path")
    else:
        print("No path found!")

    plt.title("SDF of a Circle with A* Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.legend()
    plt.savefig('fig/sdf_circle_path_plot.png', dpi=100, bbox_inches='tight')
    
if __name__ =='__main__':
    type='circle'
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)
    
    if type =='circle':
        center = (0, 0)
        radius = 1
        min_distance = 0.1
        Z = np.vectorize(lambda px, py: sdf_circle((px, py), center, radius))(X, Y)
    start = [100, 100]
    goal = [400, 400]
    start_coords = (X[start[1], start[0]], Y[start[1], start[0]])
    goal_coords = (X[goal[1], goal[0]], Y[goal[1], goal[0]])
    print("Start coordinates:", start_coords)
    print("Goal coordinates:", goal_coords)
    path = np.array(astar(start, goal, Z, min_distance))
    plot_sdf_path(X,Y,Z,path)

