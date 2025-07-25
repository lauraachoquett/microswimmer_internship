import os
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .evaluate_agent import evaluate_agent

colors = plt.cm.tab10.colors
import copy
import pickle
from statistics import mean, stdev

from scipy.spatial import KDTree

from src.env_swimmer import MicroSwimmer
from src.generate_path import (
    generate_curve_with_target_curvature,
    generate_demi_circle_path,
    generate_random_ondulating_path,
    generate_simple_line,
)
from src.plot import plot_background_velocity, plot_trajectories


def format_sci(x):
    return "{:.3e}".format(x)


def plot_robust_D(
    config_eval, file_name_or, agent, env, save_path_eval, nb_D, threshold
):
    save_path_eval_D = os.path.join(save_path_eval, "robust_D_bis/")
    if not os.path.exists(save_path_eval_D):
        os.makedirs(save_path_eval_D)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    D_values = np.linspace(0.0, 1.3 * config_eval["Dt_action"], nb_D)
    config_eval_bis = copy.deepcopy(config_eval)
    for idx_tr, thr in enumerate(threshold):
        mean_reward_D = np.zeros(nb_D)
        std_reward_D = np.zeros(nb_D)
        success_rate_D = np.zeros(nb_D)
        config_eval_bis["threshold"] = thr
        for idx, D in enumerate(D_values):
            print(f"iter : {idx}")
            config_eval_bis["D"] = D
            file_name = file_name_or + f"_{idx}"
            val = 2 * D / (1**2 * config_eval["Dt_action"])
            formatted_val = format_sci(val)
            label = r"$\frac{2D}{U^2 \Delta t}$ = " + formatted_val

            (
                rewards_per_episode,
                rewards_t_per_episode,
                rewards_d_per_episode,
                success_rate,
                states_list_per_episode,
            ) = evaluate_agent(
                agent,
                env,
                config_eval_bis["eval_episodes"],
                config_eval_bis,
                save_path_eval_D,
                file_name=file_name,
                random_parameters=False,
                label=label,
            )

            mean_reward_D[idx] = mean(rewards_per_episode)
            std_reward_D[idx] = stdev(rewards_per_episode)
            success_rate_D[idx] = success_rate

        plot_mean_reward_success_rate(
            mean_reward_D,
            std_reward_D,
            success_rate_D,
            D_values * 2 / config_eval["Dt_action"],
            r"$\frac{2D}{U^2 \Delta t}$",
            idx_tr,
            thr,
            fig,
            axs,
        )

    path_save_fig_D = os.path.join(save_path_eval_D, f"success_rate_rew_D")
    fig.suptitle(f"episodes : {config_eval['eval_episodes']}", fontsize=10)
    fig.tight_layout()
    plt.savefig(path_save_fig_D, dpi=400, bbox_inches="tight")


def plot_robust_u_bg_uniform(
    config_eval, file_name_or, agent, env, save_path_eval, nb_norm, threshold
):
    dir_d = {
        "North": np.array([0, 1]),
        "West": np.array([-1, 0]),
        "South": np.array([0, -1]),
        "East": np.array([1, 0]),
    }
    norm_values = np.linspace(0.1, 0.7, nb_norm)
    config_eval_dir = copy.deepcopy(config_eval)
    for dir, vec in dir_d.items():
        save_path_eval_dir = os.path.join(save_path_eval, dir)
        print(dir)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for idx_tr, thr in enumerate(threshold):
            config_eval_dir["threshold"] = thr
            mean_reward_D = np.zeros(nb_norm)
            std_reward_D = np.zeros(nb_norm)
            success_rate_D = np.zeros(nb_norm)
            if not os.path.exists(save_path_eval_dir):
                os.makedirs(save_path_eval_dir)
            for idx, norm in enumerate(norm_values):
                config_eval_dir["u_bg"] = vec * norm
                file_name = file_name_or + f"_{idx}"
                results = evaluate_agent(
                    agent,
                    env,
                    config_eval_dir["eval_episodes"],
                    config_eval_dir,
                    save_path_eval_dir,
                    file_name=file_name,
                    random_parameters=False,
                )
                (
                    rewards_per_episode,
                    rewards_t_per_episode,
                    rewards_d_per_episode,
                    success_rate,
                    states_list_per_episode,
                ) = results
                mean_reward = mean(rewards_per_episode)
                std_reward = stdev(rewards_per_episode)
                mean_reward_D[idx] = mean_reward
                std_reward_D[idx] = std_reward
                success_rate_D[idx] = success_rate

            plot_mean_reward_success_rate(
                mean_reward_D,
                std_reward_D,
                success_rate_D,
                norm_values,
                r"$ \frac{\|u\|}{U}$",
                idx_tr,
                thr,
                fig,
                axs,
            )
        path_save_fig_D = os.path.join(save_path_eval_dir, f"success_rate_rew_u_bg")
        fig.suptitle(f"episodes : {config_eval_dir['eval_episodes']}", fontsize=10)
        fig.tight_layout()
        plt.savefig(path_save_fig_D, dpi=400, bbox_inches="tight")


def plot_robust_u_bg_rankine(
    config_eval, file_name_or, agent, env, save_path_eval, nb_cir, threshold
):
    center = [1, 1.5]
    a = 0.5
    cir_values = np.linspace(-3, 3, nb_cir)
    config_eval_rk = copy.deepcopy(config_eval)
    save_path_eval_rk = os.path.join(save_path_eval, "rankine")
    trajectories = {}
    if not os.path.exists(save_path_eval_rk):
        os.makedirs(save_path_eval_rk)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    config_eval_rk["rankine_bg"] = True
    for idx_tr, thr in enumerate(threshold):
        config_eval_rk["threshold"] = thr
        mean_reward_D = np.zeros(nb_cir)
        std_reward_D = np.zeros(nb_cir)
        success_rate_D = np.zeros(nb_cir)
        if not os.path.exists(save_path_eval_rk):
            os.makedirs(save_path_eval_rk)
        for idx, cir in enumerate(cir_values):
            file_name = file_name_or + f"_{idx}"
            parameters = [center, a, cir]

            result = evaluate_agent(
                agent,
                env,
                config_eval_rk["eval_episodes"],
                config_eval_rk,
                save_path_eval_rk,
                file_name=file_name,
                random_parameters=False,
                parameters=parameters,
            )
            (
                rewards_per_episode,
                rewards_t_per_episode,
                rewards_d_per_episode,
                success_rate,
                states_list_per_episode,
            ) = result
            trajectories[f"thr_{thr}_cir_{cir}"] = result

            mean_reward = mean(rewards_per_episode)
            std_reward = stdev(rewards_per_episode)
            mean_reward_D[idx] = mean_reward
            std_reward_D[idx] = std_reward
            success_rate_D[idx] = success_rate

        plot_mean_reward_success_rate(
            mean_reward_D,
            std_reward_D,
            success_rate_D,
            cir_values * (1 / 2 * a * np.pi),
            r"$\frac{\Gamma}{2 \pi a U}$",
            idx_tr,
            thr,
            fig,
            axs,
        )
        path_save_fig_D = os.path.join(save_path_eval_rk, f"success_rate_rew_u_bg_rk")
        fig.suptitle(f"episodes : {config_eval_rk['eval_episodes']}", fontsize=10)
        fig.tight_layout()
        plt.savefig(path_save_fig_D, dpi=400, bbox_inches="tight")
        pkl_save_path = os.path.join(
            save_path_eval_rk, f"{file_name_or}_trajectories.pkl"
        )
        with open(pkl_save_path, "wb") as f:
            pickle.dump(trajectories, f)


def plot_mean_reward_success_rate(
    mean_rewards,
    std_rewards,
    list_success_rate,
    abscisse,
    xlabel,
    idx_tr,
    thr,
    fig,
    axs,
):
    axs[0].plot(
        abscisse,
        mean_rewards,
        color=colors[idx_tr],
        label=r"$\delta {:.3f}$".format(thr),
    )
    axs[0].fill_between(
        abscisse,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color=colors[idx_tr],
    )
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("Reward")
    axs[0].legend()

    axs[1].plot(
        abscisse,
        list_success_rate,
        color=colors[idx_tr],
        label=r"$\delta {:.3f}$".format(thr),
    )
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Success rate")
    axs[1].legend()


def visualize_streamline(
    agent,
    config_eval,
    file_name_or,
    save_path_eval,
    type="",
    title="",
    k=0,
    parameters=[],
    offset=0.2,
):
    save_path_streamline = os.path.join(save_path_eval, "streamlines/")
    if not os.path.exists(save_path_streamline):
        os.makedirs(save_path_streamline)
    trajectories = {}
    config_eval_v = copy.deepcopy(config_eval)
    if type == "line":
        config_eval_v["p_target"] = config_eval_v["p_target"] / 4
        p_target = config_eval_v["p_target"]
    else:
        p_target = config_eval_v["p_target"]
    p_0 = config_eval_v["p_0"]
    nb_points_path = config_eval_v["nb_points_path"]

    nb_starting_point = 20
    p_0_above = p_0 + np.array([0, offset])
    p_target_above = p_target + np.array([0, offset])
    p_0_below = p_0 + np.array([0, -offset])
    p_target_below = p_target + np.array([0, -offset])
    if type == "ondulating":
        path = generate_random_ondulating_path(
            p_0, p_target, nb_points_path, amplitude=0.5, frequency=2
        )
        path_above_point = generate_random_ondulating_path(
            p_0_above, p_target_above, nb_starting_point, amplitude=0.5, frequency=2
        )
        path_below_point = generate_random_ondulating_path(
            p_0_below, p_target_below, nb_starting_point, amplitude=0.5, frequency=2
        )
    if type == "circle":
        path, _ = generate_demi_circle_path(p_0, p_target, nb_points_path)
        path_above_point, _ = generate_demi_circle_path(
            p_0_above, p_target_above, nb_starting_point
        )
        path_below_point, _ = generate_demi_circle_path(
            p_0_below, p_target_below, nb_starting_point
        )
    if "curve" in type:
        path = generate_curve(p_0, p_target, k, nb_points_path)
        path_above_point = generate_curve(
            p_0_above, p_target_above, k, nb_starting_point
        )
        path_below_point = generate_curve(
            p_0_below, p_target_below, k, nb_starting_point
        )
    if type == "line":
        path, _ = generate_simple_line(p_0, p_target, nb_points_path)
        path_above_point, _ = generate_simple_line(
            p_0_above, p_target_above, nb_starting_point
        )
        path_below_point, _ = generate_simple_line(
            p_0_below, p_target_below, nb_starting_point
        )

    config_eval_v["path"] = path
    config_eval_v["tree"] = KDTree(path)
    config_eval_v["D"] = 0
    path_above_point = path_above_point[:-1]
    path_below_point = path_below_point[:-1]
    path_starting_point = np.concatenate((path_above_point, path_below_point), axis=0)
    file_name_or += title
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(path[:, 0], path[:, 1], label="path", color="black", linewidth=2, zorder=10)

    trajectories["path"] = path
    for starting_point in path_starting_point:
        config_eval_v["x_0"] = starting_point
        env = MicroSwimmer(
            config_eval_v["x_0"],
            config_eval_v["C"],
            config_eval_v["Dt_action"] / config_eval_v["steps_per_action"],
            config_eval_v["velocity_bool"],
            config_eval_v["n_lookahead"],
            config_eval_v["velocity_ahead"],
            config_eval_v["add_action"],
        )
        _, _, _, _, states_list_per_episode = evaluate_agent(
            agent=agent,
            env=env,
            eval_episodes=1,
            config=config_eval_v,
            save_path_result_fig=save_path_streamline,
            file_name=file_name_or,
            random_parameters=False,
            title="",
            plot=False,
            parameters=parameters,
            plot_background=False,
        )
        trajectories[f"{starting_point}"] = states_list_per_episode
        plot_trajectories(ax, states_list_per_episode, path, title="streamlines")
    pkl_save_directory = os.path.join(save_path_streamline, type)
    os.makedirs(pkl_save_directory, exist_ok=True)
    pkl_save_path = os.path.join(pkl_save_directory, f"{file_name_or}_trajectories.pkl")
    with open(pkl_save_path, "wb") as f:
        pickle.dump(trajectories, f)

    # x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    # x_range = x_max - x_min
    # y_range = y_max - y_min
    # max_range = max(x_range, y_range)
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    # ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_min - offset, y_max + offset)

    ax.set_aspect("equal")
    if config_eval_v["uniform_bg"]:
        x_bound = ax.get_xlim()
        y_bound = ax.get_ylim()
        dir, norm = parameters
        type = "uniform"
        plot_background_velocity(type, x_bound, y_bound, dir=dir, norm=norm)
    elif config_eval_v["rankine_bg"]:
        x_bound = ax.get_xlim()
        y_bound = ax.get_ylim()
        center, a, cir = parameters
        type = "rankine"
        plot_background_velocity(type, x_bound, y_bound, a=a, center=center, cir=cir)
    directory_fig = os.path.join(save_path_streamline, type)
    os.makedirs(directory_fig, exist_ok=True)
    path_save_fig = os.path.join(directory_fig, file_name_or + ".png")
    fig.savefig(path_save_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_streamlines_beta(agents_file, dict):
    viridis = cm.get_cmap("viridis", len(agents_file))
    colors = sns.color_palette("Set2")

    beta_values = np.array([0.05, 0.25, 0.4])
    save_path_eval = "fig/"
    fig, ax = plt.subplots(figsize=(8, 4))
    added_labels = set()  # Track labels already added to avoid duplicates
    j = 0
    for id, agent_file in enumerate(agents_file):
        path_config = os.path.join(agent_file, "config.pkl")
        with open(path_config, "rb") as f:
            config = pickle.load(f)
        if config["beta"] in beta_values:
            print(path_config)
            for i, key in enumerate(dict.keys()):
                file_name_or = f"streamline_{key}_line_trajectories.pkl"
                path_trajectories = os.path.join(
                    agent_file, "eval_bg/streamlines/line/", file_name_or
                )
                with open(path_trajectories, "rb") as f:
                    trajectories = pickle.load(f)
                path = trajectories["path"]
                for idx, (traj_key, trajectory) in enumerate(trajectories.items()):
                    if (
                        traj_key != "path" and idx % 4 == 0
                    ):  # Take every other trajectory
                        label = (
                            f"{config['beta']}"
                            if f"{config['beta']}" not in added_labels
                            else None
                        )
                        if label:
                            added_labels.add(label)
                        plot_trajectories(
                            ax,
                            trajectory,
                            path,
                            title="streamlines",
                            color_id=j,
                            colors=None,
                            label=label,
                        )
            j += 1
    ax.plot(path[:, 0], path[:, 1], color="black", linewidth=2)
    ax.set_ylim([-0.05, 0.05])
    # ax.set_aspect("equal")
    # Place the legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    path_save_comparison = os.path.join(
        save_path_eval, f"comparison_streamlines_beta_2"
    )
    fig.savefig(path_save_comparison, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    agent_name = []
    visualize_streamline(
        agent,
        config_eval,
        file_name_or,
        save_path_eval,
        type="",
        title="",
        k=0,
        parameters=[],
        offset=0.2,
    )
