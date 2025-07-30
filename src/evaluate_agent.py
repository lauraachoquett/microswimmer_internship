import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from src.frenet import compute_frenet_frame, double_reflection_rmf
from src.simulation import rankine_vortex, uniform_velocity
from src.utils import random_bg_parameters

colors = plt.cm.tab10.colors
import copy

from src.generate_path import generate_curve
from src.plot import (plot_html_3d, plot_trajectories, plot_trajectories_3D,
                      video_trajectory,paraview_export)


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
    video=False,
    D_state=None
):
    config = copy.deepcopy(config)
    parameters = copy.deepcopy(parameters)

    dim = config["dim"]

    p_0 = config["p_0"]
    p_target = config["p_target"]

    t_max = config["t_max"]
    steps_per_action = config["steps_per_action"]
    Dt_action = config["Dt_action"]
    Dt_sim = Dt_action / steps_per_action
    beta = config["beta"]
    gamma = config['gamma'] if 'gamma' in config else 0.000
    pow_d = config['pow_d'] if 'pow_d' in config else 1
    D = config["D"]
    threshold = config["threshold"]
    x = config["x_0"]
    action = np.zeros(3)

    iter = 0
    episode_num = 0
    episode_reward = 0
    episode_rew_t = 0
    episode_rew_d = 0
    rewards_per_episode = []
    rewards_t_per_episode = []
    rewards_d_per_episode = []
    x_pos_episode = []
    x_pos_list_per_episode = []
    state_episode = []
    action_list=[]
    action_list_per_episode=[]
    count_succes = 0
    v_hist = []

    u_bg = np.zeros(dim)
    velocity_func = lambda x: u_bg

    type = ""

    if random_parameters:
        dir, norm, center, a, cir = random_bg_parameters()
    else:
        if len(parameters) > 0:
            if config["uniform_bg"]:
                dir, norm = parameters
                dir = np.array(dir)
                type = "uniform"
                center, a, cir = np.zeros(dim), 0, 0
            if config["rankine_bg"]:
                type = "rankine"
                center, a, cir = parameters
                dir, norm = np.zeros(dim), 0
        else:
            dir, norm, center, a, cir = np.zeros(dim), 0, np.zeros(dim), 0, 0

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
        velocity_func = lambda x: rankine_vortex(x, a, center, cir,dim)

    if list_of_path_tree is not None:
        path, tree = list_of_path_tree[0]
        nb_of_path = len(list_of_path_tree)
    else:
        list_of_path_tree = [[config["path"], config["tree"]]]
        path, tree = list_of_path_tree[0]
        nb_of_path = 1

    #### EVALUATION ####
    T, N, B = double_reflection_rmf(path)
    state, done = env.reset(x,tree, path, T, velocity_func, N, B), False
    while episode_num < eval_episodes:
        x_pos_episode.append(x)
        state_episode.append(state)

        iter += 1

        if iter == 1 : 
            past_action=action
            action = agent.select_action(state)
            action_list.append(list(action))
            
        if iter % steps_per_action == 0 :
            past_action = action
            action = agent.select_action(state)
            # print("State :",state)
            # print("Action :",action)
            action_list.append(list(action))


        if config["uniform_bg"] or config["rankine_bg"]:
            u_bg = velocity_func(x)

        if velocity_func is not None:
            u_bg = velocity_func(x)
            v = np.linalg.norm(u_bg)

        next_state, reward, done, info = env.step(
            action=action,
            past_action = past_action,
            tree=tree,
            path=path,
            T=T,
            x_target=p_target,
            beta=beta,
            gamma=gamma,
            pow_d=pow_d,
            D=D,
            u_bg=u_bg,
            threshold=threshold,
            sdf=sdf,
            N=N,
            B=B,
            D_state=D_state,
        )

        x = info["x"]
        episode_rew_t += info["rew_t"]
        episode_rew_d += info["rew_d"]
        episode_reward += reward
        state = next_state

        if done or iter * Dt_sim > t_max:
            if done:
                count_succes += 1
            x_pos_list_per_episode.append([np.array(x_pos_episode), iter])
            action_list_per_episode.append(action_list)
            iter = 0
            episode_num += 1
            x_pos_episode = []
            action_list = []
            rewards_per_episode.append(episode_reward)
            rewards_t_per_episode.append(episode_rew_t)
            rewards_d_per_episode.append(episode_rew_d)
            episode_reward = 0
            episode_rew_t = 0
            episode_rew_d = 0

            ## Other path ##
            path, tree = list_of_path_tree[episode_num % nb_of_path]
            p_0 = path[0]
            p_target = path[-1]
            x = p_0
            T, N, B = double_reflection_rmf(path)
            ## Reset ##
            state, done = env.reset(x,tree, path, T, velocity_func, N, B), False

    if video:
        print("Making video...")
        path_save_video = os.path.join(
            save_path_result_fig, file_name + "_video_trajectory.mp4"
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        if obstacle_contour is not None:
            ax.scatter(
                obstacle_contour[:, 0], obstacle_contour[:, 1], color="black", s=0.2
            )
        path = list_of_path_tree[0][0]
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
            x_pos_list_per_episode[0][0],
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
        if len(x_pos_list_per_episode) >= 4:
            trajectories = x_pos_list_per_episode[-4:]
        else:
            trajectories = x_pos_list_per_episode
        path_save_fig = os.path.join(save_path_result_fig, file_name)
        save_path_html = os.path.join(save_path_result_fig, file_name + "_3D.html")
        save_path_paraview = os.path.join(save_path_result_fig, file_name + "_paraview/")
        if dim == 2:
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
                    color="balck",
                    linewidth=1,
                    zorder=0,
                )

            ylim = ax.get_ylim()
            if ylim[1] - ylim[0] < 1 / 3:
                ax.set_ylim(top=1.0, bottom=-1)

            plot_trajectories(
                ax,
                trajectories,
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
            ax = fig.add_subplot(111, projection="3d")

            for elt in list_of_path_tree:
                path, _ = elt
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color="black", linewidth=1)

            plot_trajectories_3D(
                ax,
                trajectories,
                title=title,
                type=type,
                a=a,
                cir=cir,
                norm=norm,
            )

            # Si ax.set_axis_off() ne fonctionne pas pour un plot 3D, tu peux commenter cette ligne
            ax.set_axis_off()

        fig.savefig(path_save_fig, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if dim == 3:
            list_of_path = [x[0] for x in list_of_path_tree]
            plot_html_3d(trajectories, save_path_html, list_of_path,dir,center, a,obstacle_contour)
            if config['paraview']:
                paraview_export(path,  save_path_paraview,trajectories)

        os.makedirs(path_save_fig, exist_ok=True)
        state_episode = np.array(state_episode)
        n = state_episode.shape[1]
        states_reshaped = state_episode.reshape(-1, n, 3)
        norms = np.linalg.norm(states_reshaped, axis=2)  # shape: (n_steps, n)

        fig, axs = plt.subplots(2, 4, figsize=(15, 6))
        axs = axs.flatten()

        for i in range(n):
            if i ==1:
                min_norm = norms[:, i].min()
                max_norm = 5
            else :
                min_norm = norms[:, i].min()
                max_norm = norms[:, i].max()
            axs[i].hist(norms[:, i], bins=40, edgecolor='black', range=(min_norm, max_norm))
            axs[i].set_title(f"Vecteur {i}")
            axs[i].set_xlabel("Norme")
            axs[i].set_ylabel("Fréquence")
            axs[i].set_xlim(min_norm, max_norm)

        plt.tight_layout()
        plt.savefig(os.path.join(path_save_fig, 'hist_norm.png'))
        plt.close(fig)
        
    if len(x_pos_list_per_episode) >= 4:
        trajectories = x_pos_list_per_episode[-4:]
    else:
        trajectories = x_pos_list_per_episode
        
    if len(x_pos_list_per_episode) >= 4:
        actions = action_list_per_episode[-4:]
    else:
        actions = action_list_per_episode
        
    return (
        rewards_per_episode,
        rewards_t_per_episode,
        rewards_d_per_episode,
        count_succes / eval_episodes,
        trajectories,
        actions
    )
