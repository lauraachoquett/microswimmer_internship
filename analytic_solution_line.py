import numpy as np
from generate_path import generate_simple_line
import matplotlib.pyplot as plt
from math import atan2

def f(v, v_target, beta):
    return np.linalg.norm(v - v_target) + beta * abs(v[1])


def find_next_v(v_t, v_target, beta, Dt, num_angles=120):
    dir = v_target - v_t
    theta = atan2(dir[1],dir[0])
    alpha = np.pi/8
    best_v = None
    best_f = np.inf
    best_dir = np.zeros(2)
    for angle in np.linspace(theta - alpha, theta + alpha, num_angles):
        candidate = v_t + Dt * np.array([np.cos(angle), np.sin(angle)])
        val = f(candidate, v_target, beta)
        if val < best_f:
            best_f = val
            best_v = candidate
            best_dir = np.array([np.cos(angle), np.sin(angle)])
    return best_v,best_dir




if __name__ == '__main__':
    v_0 = np.array([1/2, 1/4])
    p_0 = np.array([0.0, 0])
    p_target = np.array([2, 0])
    beta = 0.25
    Dt = 0.001
    U = 1

    T = 50  
    vs = [v_0]

    for t in range(T):
        while np.linalg.norm(vs[-1]-p_target)>0.05:
            v_t = vs[-1]
            v_next,_ = find_next_v(v_t, p_target, beta, Dt)
            vs.append(v_next)

    vs = np.array(vs)
    print(vs)

    path,_ = generate_simple_line(p_0,p_target,100)
    plt.plot(path[:, 0], path[:, 1],label='path', color='black', linewidth=2)
    plt.plot(vs[:, 0], vs[:, 1], label='Trajectoire')
    plt.plot(p_target[0], p_target[1], 'rx', label='Cible')
    plt.axis('equal')
    plt.legend()
    plt.title("Évolution de v_t")
    plt.show()