from math import atan2, cos, exp, pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from scipy.interpolate import CubicSpline


def compute_curvature(path,n_points):
    t_vals= np.linspace(0, 1, n_points)
    x = path[:, 0]
    y = path[:, 1]

    cs_x = CubicSpline(t_vals, x, bc_type="natural")
    cs_y = CubicSpline(t_vals, y, bc_type="natural")

    dx = cs_x(t_vals, 1)
    dy = cs_y(t_vals, 1)
    ddx = cs_x(t_vals, 2)
    ddy = cs_y(t_vals, 2)

    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def length_path(path):
    distances = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    return distances

def generate_simple_line(p_0, p_target, nb_points):
    t = np.linspace(0, 1, nb_points)
    path = p_0 * t[:, None] + (1 - t)[:, None] * p_target
    d = np.linalg.norm((p_target - p_0))
    return np.flip(path, axis=0), d


def generate_line_two_part(p_0, p_1, p_target, nb_points):
    t = np.linspace(0, 1, nb_points)
    path_1 = p_0 * t[:, None] + (1 - t)[:, None] * p_1
    path_1 = np.flip(path_1, axis=0)
    path_2 = np.flip(p_1 * t[:, None] + (1 - t)[:, None] * p_target, axis=0)
    path = np.concatenate((path_1, path_2), axis=0)
    d = np.linalg.norm(p_1 - p_0) + np.linalg.norm(p_target - p_0)
    return path, d


def generate_demi_circle_path(p_0, p_target, nb_points):
    p_0 = np.array(p_0)
    p_target = np.array(p_target)

    dir_vec = p_target - p_0
    length = np.linalg.norm(dir_vec)
    r = length / 2
    center = p_0 + dir_vec / 2

    alpha = np.arctan2(dir_vec[1], dir_vec[0])
    thetas = np.linspace(0, np.pi, nb_points)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    circle_points = np.stack([x, y], axis=1)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_points = circle_points @ R.T
    path = center + rotated_points
    d = pi * r
    return np.flip(path, axis=0), d


def generate_curve_with_target_curvature(p_0, p_target, kappa_max, nb_points):
    p_0 = np.array(p_0)
    p_target = np.array(p_target)
    delta = p_target - p_0
    L_x = delta[0]
    L_y = delta[1]

    norm_squared = L_x**2 + L_y**2
    k = - (kappa_max * norm_squared**(3/2)) / (2 * L_x**2)

    t = np.linspace(0, 1, nb_points)
    x = (1 - t) * p_0[0] + t * p_target[0]
    y = (1 - t) * p_0[1] + t * p_target[1] + k * t * (1 - t) * (p_target[0] - p_0)[0]

    path = np.stack([x, y], axis=1)
    return path



def generate_random_ondulating_path(p_0, p_target, n_points=1000, kappa_max=0.5, frequency=5):
    t = np.linspace(0, 1, n_points)
    direction = np.array(p_target) - np.array(p_0)
    length = np.linalg.norm(direction)  # longueur du segment entre p0 et p_target
    direction_unit = direction / length

    # vecteur orthogonal à la direction (pour l'ondulation)
    ortho = np.array([-direction_unit[1], direction_unit[0]])

    # ⚠️ Amplitude corrigée avec la vraie échelle
    amplitude = (kappa_max * length**2) / (2 * np.pi * frequency)**2

    # base : interpolation linéaire entre p0 et p_target
    base = np.outer(t, direction) + np.array(p_0)
    # perturbation sinusoïdale
    deviation = amplitude * np.sin(2 * np.pi * frequency * t)
    path = base + np.outer(deviation, ortho)
    print("Max curvature : ",np.max(compute_curvature(path,n_points)))

    return path

def generate_random_ondulating_path_old(
    p_0, p_target, n_points=100, max_curvature=1.0, amplitude=0.1, frequency=5
):
    t = np.linspace(0, 1, n_points)
    x_base = np.linspace(p_0[0], p_target[0], n_points)
    y_base = np.linspace(p_0[1], p_target[1], n_points)
    noise = amplitude * np.sin(frequency * 2 * np.pi * t)
    y_ondulating = y_base + noise
    cs = CubicSpline(t, np.column_stack([x_base, y_ondulating]), bc_type="natural")
    path = cs(t)
    print("Max curvature : ",np.max(compute_curvature(path,n_points)))
    return path
    
def func_k_max(A, N, f, n):
    if n > N:
        amp = 1 - exp(1)
        freq = f / 16
    else:
        amp = 1 - exp(n / N)
        freq = f / (8 * (1 + (n / N) ** (3 / 2)))
    return -A * amp * cos(n * freq)


def plot_path(p_0, p_target, nb_points, type="line"):
    if type == "line":
        path, _ = generate_simple_line(p_0, p_target, nb_points)
    if type == "two_lines":
        p_1 = [1 / 2, 1]
        path = generate_line_two_part(p_0, p_1, p_target, nb_points)
    if type == "circle":
        path, _ = generate_demi_circle_path(p_0, p_target, nb_points)
    if type == "curve":
        nb_points = 5000
        k = 1.71
        path = generate_curve_with_target_curvature(p_0, p_target, k, nb_points)
        print("Max curvature : ",np.max(compute_curvature(path,nb_points)))
        
    if type == "ondulating_path_hard":
        path = generate_random_ondulating_path(
            p_0, p_target, n_points=5000, kappa_max=17, frequency=2
        )
        print("OLD")
        path = generate_random_ondulating_path_old(
            p_0, p_target, n_points=5000, amplitude=0.5, frequency=2
        )
    print("Length : ", np.sum(length_path(path) ))
    
    plt.close()
    plt.plot(path[:, 0], path[:, 1], label="path")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(p_0[0], p_0[1], color="red", label="Starting point")
    plt.scatter(p_target[0], p_target[1], color="blue", label="Ending point")
    plt.legend()
    plt.title(f"Path : {type}")
    plt.savefig(f"fig/path_{type}.png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    p_0 = np.array([0, 0])
    p_target = np.array([2, 0])
    nb_points = 200
    # plot_path(p_0,p_target,nb_points,'line')
    A = 1
    N = 400
    f = 2
    n_values = np.linspace(1, 700, 700, dtype=int)
    output = [func_k_max(A, N, f, n) for n in n_values]
    nb_points = 700
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    plt.plot(n_values, 2*np.array(output))
    colors = plt.cm.viridis(np.linspace(0, 1, 20))
    for i in range(5):
        n = 270 + i*2
        k = func_k_max(A, N, f, n)
        plt.scatter(n, 2*k, color=colors[n % 20])

    plt.ylabel(r"$\kappa \cdot L$")
    plt.xlabel(r"Number of episodes")
    plt.subplot(1, 2, 2)
    for i in range(5):
        n = 270 + i*2
        k = func_k_max(A, N, f, n)
        path = generate_curve_with_target_curvature(p_0, p_target, k, nb_points)
        plt.plot(path[:, 0], path[:, 1], label=r"$\kappa \cdot L$" + f" : {2*k:.2f}", color=colors[n % 20])

    plt.scatter(p_0[0], p_0[1], color="black")
    plt.scatter(p_target[0], p_target[1], color="black")
    plt.axis(False)
    plt.legend()
    plt.savefig("fig/smooth_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    p_target=[2, 0]
    p_0=[0, 0]
    plot_path(p_0, p_target, nb_points, type="curve")
