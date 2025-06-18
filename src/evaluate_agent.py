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
    D = config["D"]
    threshold = config["threshold"]
    x = config["x_0"]
    

    iter = 0
    episode_num = 0
    episode_reward = 0
    episode_rew_t = 0
    episode_rew_d = 0
    rewards_per_episode = []
    rewards_t_per_episode = []
    rewards_d_per_episode = []
    states_episode = []
    states_list_per_episode = []
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
        states_episode.append(x)
        iter += 1

        if iter % steps_per_action == 0 or iter == 1:
            action = agent.select_action(state)
            action_list.append(list(action))


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
            T=T,
            x_target=p_target,
            beta=beta,
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
            states_list_per_episode.append([np.array(states_episode), iter])
            action_list_per_episode.append(action_list)
            iter = 0
            episode_num += 1
            states_episode = []
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
        if len(states_list_per_episode) >= 4:
            trajectories = states_list_per_episode[-4:]
        else:
            trajectories = states_list_per_episode
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

    # print(mean(v_hist))
    # path_save_fig = os.path.join(save_path_result_fig, file_name + "_hist_v.png")
    # plt.hist(v_hist, bins=50, color="blue", alpha=0.7)
    # plt.axvline(
    #     mean(v_hist), color="green", linestyle="dashed", linewidth=1.5, label="Mean"
    # )
    # plt.xlabel(r"$u_{bg} / \|U\|$")
    # plt.legend()
    # plt.savefig(path_save_fig, dpi=100, bbox_inches="tight")
    # plt.close()
    
    if len(states_list_per_episode) >= 4:
        trajectories = states_list_per_episode[-4:]
    else:
        trajectories = states_list_per_episode
        
    if len(states_list_per_episode) >= 4:
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
