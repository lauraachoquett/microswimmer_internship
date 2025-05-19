import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from src.simulation import rankine_vortex, uniform_velocity
from src.utils import random_bg_parameters

colors = plt.cm.tab10.colors
import copy

from src.generate_path import generate_curve
from src.plot import plot_trajectories,video_trajectory


def evaluate_agent(
    agent,
    env,
    eval_episodes,
    config,
    save_path_result_fig,
    file_name,
    random_parameters,
    list_of_path_tree=None,
    title="",
    plot=True,
    parameters=[],
    plot_background=False,
    rng=None,
    obstacle_contour=None,
    sdf=None,
    velocity_func_l=None,
    video=False
):
    dim=2
    config = copy.deepcopy(config)
    parameters = copy.deepcopy(parameters)
    rewards_per_episode = []
    t_max = config["t_max"]
    p_target = config["p_target"]
    steps_per_action = config["steps_per_action"]
    t_max = config["t_max"]
    Dt_action = config["Dt_action"]
    Dt_sim = Dt_action / steps_per_action
    p_0 = config["p_0"]
    beta = config["beta"]
    iter = 0
    episode_num = 0
    episode_reward = 0
    episode_rew_t = 0
    episode_rew_d = 0
    rewards_t_per_episode = []
    rewards_d_per_episode = []

    states_episode = []
    states_list_per_episode = []
    u_bg = config["u_bg"]
    D = config["D"]
    type = ""
    threshold = config["threshold"]
    x = config["x_0"]
    count_succes = 0
    v_hist = []
    if random_parameters:
        dir, norm, center, a, cir = random_bg_parameters()
    else:
        if len(parameters) > 0:
            if config["uniform_bg"]:
                dir, norm = parameters
                dir = np.array(dir)
                type = "uniform"
                center, a, cir = np.zeros(2), 0, 0
            if config["rankine_bg"]:
                type = "rankine"
                center, a, cir = parameters
                dir, norm = np.zeros(2), 0
        else:
            dir, norm, center, a, cir = np.zeros(2), 0, np.zeros(2), 0, 0

    if u_bg.any() != 0:
        type = "uniform"
        norm = np.linalg.norm(u_bg)
        dir = np.array(u_bg / norm)
        plot_background = True
    velocity_func = None
    if velocity_func_l is not None:
        velocity_func = lambda x: velocity_func_l(x).squeeze()

    if config["uniform_bg"]:
        velocity_func = lambda x: uniform_velocity(dir, norm)
    elif config["rankine_bg"]:
        velocity_func = lambda x: rankine_vortex(x, a, center, cir)

    if list_of_path_tree is not None:
        path, tree = list_of_path_tree[0]
        nb_of_path = len(list_of_path_tree)
    else:
        list_of_path_tree = [[config["path"], config["tree"]]]
        path, tree = list_of_path_tree[0]
        nb_of_path = 1

    #### EVALUATION ####
    state, done = env.reset(tree, path, velocity_func), False
    while episode_num < eval_episodes:
        states_episode.append(x)
        iter += 1

        if iter % steps_per_action == 0 or iter == 1:
            action = agent.select_action(state)

        if config["uniform_bg"] or config["rankine_bg"]:
            u_bg = velocity_func(x)

        if velocity_func is not None:
            u_bg = velocity_func(x)
            v = np.linalg.norm(u_bg)
            v_hist.append(v)

        next_state, reward, done, info = env.step(
            action=action,
            tree=tree,
            path=path,
            x_target=p_target,
            beta=beta,
            D=D,
            u_bg=u_bg,
            threshold=threshold,
            sdf=sdf,
        )

        x = info["x"]
        episode_rew_t += info["rew_t"]
        episode_rew_d += info["rew_d"]
        episode_reward += reward
        state = next_state

        if done or iter * Dt_sim > t_max:
            if done:
                count_succes += 1
            states_list_per_episode.append([np.array(states_episode), iter])
            state, done = env.reset(tree, path, velocity_func), False
            iter = 0
            episode_num += 1
            x = p_0
            states_episode = []
            rewards_per_episode.append(episode_reward)
            rewards_t_per_episode.append(episode_rew_t)
            rewards_d_per_episode.append(episode_rew_d)
            episode_reward = 0
            episode_rew_t = 0
            episode_rew_d = 0
            path, tree = list_of_path_tree[episode_num % nb_of_path]
    if video : 
        print("Making video...")
        path_save_video = os.path.join(save_path_result_fig,file_name+"_video_trajectory.mp4")
        fig, ax = plt.subplots(figsize=(10, 8))
        if obstacle_contour is not None:
            ax.scatter(
                obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.2
            )
        for elt in list_of_path_tree:
            path, _ = elt
            ax.plot(
                path[:, 0],
                path[:, 1],
                label="path",
                color="firebrick",
                linewidth=1,
                zorder=0,
            )
        ylim = ax.get_ylim()
        if ylim[1] - ylim[0] < 1 / 3:
            ax.set_ylim(top=1.0, bottom=-1)
        video_trajectory(
            fig,
            ax,
            states_list_per_episode[0][0],
            path,
            title,
            a,
            center,
            cir,
            dir,
            norm,
            plot_background,
            path_save_video,
        )

    if plot:
        path_save_fig = os.path.join(save_path_result_fig, file_name)

        if dim == 2:
            fig, ax = plt.subplots(figsize=(10, 8))

            if obstacle_contour is not None:
                ax.scatter(obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.2)

            for elt in list_of_path_tree:
                path, _ = elt
                ax.plot(path[:, 0], path[:, 1], label="path", color="black", linewidth=1, zorder=0)

            ylim = ax.get_ylim()
            if ylim[1] - ylim[0] < 1 / 3:
                ax.set_ylim(top=1.0, bottom=-1)

            plot_trajectories(
                ax,
                states_list_per_episode[-4:],
                path,
                title,
                a,
                center,
                cir,
                dir,
                norm,
                plot_background,
                type=type,
            )

            ax.set_aspect("equal")
            ax.set_axis_off()

        elif dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for elt in list_of_path_tree:
                path, _ = elt
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color="black", linewidth=2)

            plot_trajectories_3D(
                ax,
                states_list_per_episode[-4:],
                title=title,
                type=type,
                a=a,
                cir=cir,
                norm=norm,
            )
            

        ax.set_aspect("equal")
        ax.set_axis_off()

        fig.savefig(path_save_fig, dpi=400, bbox_inches="tight")
        plt.close(fig)
        
        
        
    if dim == 3:
        path_save_html =  os.path.join(save_path_result_fig, "trajectoires_3d.html")
        plot_interactif(path,states_list_per_episode[-4:],path_save_html)

    return (
        rewards_per_episode,
        rewards_t_per_episode,
        rewards_d_per_episode,
        count_succes / eval_episodes,
        states_list_per_episode[-4:],
    )
