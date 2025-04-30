from math import atan2, cos, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np


def dW(delta_t, rng=None):
    return rng.normal(loc=0.0, scale=np.sqrt(delta_t), size=(2))


def solver(x, U, p, Dt, D, u_bg=np.zeros(2), rng=None, bounce_thr=0.0, sdf=None):
    if rng is None:
        rng = np.random.default_rng()
    dW_t = dW(Dt, rng)
    next_x = x + u_bg * Dt + U * p * Dt + sqrt(D) * dW_t

    if sdf is not None:
        if sdf(next_x) > bounce_thr:
            return x
    return next_x


def rankine_vortex(pos, a, center, cir):
    x, y = pos
    cx, cy = center
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    if r == 0:
        return np.array([0.0, 0.0])
    v_r = 0
    if r <= a:
        v_theta = cir / (2 * np.pi) * (r / a**2)
    else:
        v_theta = cir / (2 * np.pi) * (1 / r)
    alpha = np.arctan2(dy, dx)
    v_x = -v_theta * np.sin(alpha)
    v_y = v_theta * np.cos(alpha)
    return np.array([v_x, v_y])


def uniform_velocity(dir, norm):
    return dir * norm


def run_sde(nb_steps, t_init, t_end, U=1, p=np.ones(2), x_0=np.zeros(2)):
    D = 0.05

    Dt = (t_end - t_init) / nb_steps

    traj = np.zeros((nb_steps, 2))
    traj[0] = x_0

    traj = np.zeros((nb_steps, 2))
    traj[0] = np.array([-1, -1])

    for n in range(nb_steps - 1):
        traj[n + 1] = solver(traj[n], U, p, Dt, D)
    return traj


def plot_simulation(num_sims, nb_steps, t_init, t_end):

    for i in range(num_sims):
        traj = run_sde(nb_steps, t_init, t_end)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Sim : {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectories with {nb_steps} steps for {t_end - t_init} (time) ")
    plt.legend()
    plt.savefig("fig/SDE_test.png", dpi=100, bbox_inches="tight")


def brownian(nb_sim):
    x_final = np.zeros(nb_sim)
    for i in range(nb_sim):
        traj = run_sde(100, 0, 1, 0, 0.5)
        x_final[i] = np.linalg.norm(traj[-1]) ** 2
    print(np.mean(x_final) / 2)


if __name__ == "__main__":
    num_sims = 5
    nb_steps = 100
    t_init = 0
    t_end = 2
    plot_simulation(num_sims, nb_steps, t_init, t_end)
