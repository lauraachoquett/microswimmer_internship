import argparse
import os
import pickle
from datetime import datetime
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree
import shutil

import src.TD3 as TD3
from src.env_swimmer import MicroSwimmer
from src.evaluate_agent import evaluate_agent
from src.generate_path import *
from src.invariant_state import *
from src.simulation import rankine_vortex, uniform_velocity
from src.utils import ReplayBuffer, courbures, random_bg_parameters

colors = plt.cm.tab10.colors
import copy
import json
import random
from statistics import mean


from src.visualize import visualize_streamline


def format_sci(x):
    return "{:.3e}".format(x)


def load_agent_config(agent, eval_freq, random_curve, uniform_bg, rankine_bg):
    policy_file = os.path.join(config["load_model"], "models/agent")
    agent.load(policy_file)
    path_config = os.path.join(config["load_model"], "config.pkl")
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    print("Policy loaded !")
    config["episode_start_update"] = eval_freq * 2
    config["episode_update"] = 2
    config["random_curve"] = random_curve
    config["uniform_bg"] = uniform_bg
    config["rankine_bg"] = rankine_bg
    return agent, config


def varying_curve_init(config):
    print("Random curve")

    p_0 = config["p_0"]
    p_target = config["p_target"]
    curve_path_plus = generate_curve_with_target_curvature(p_0, p_target, 1, config["nb_points_path"])
    curve_path_minus = generate_curve_with_target_curvature(p_0, p_target, -1, config["nb_points_path"])
    curve_tree_plus = KDTree(curve_path_plus)
    curve_tree_minus = KDTree(curve_path_minus)
    list_of_path_tree = [
        [curve_path_plus, curve_tree_plus],
        [curve_path_minus, curve_tree_minus],
    ]
    line_path, _ = generate_simple_line(p_0, p_target, config["nb_points_path"])
    line_tree = KDTree(line_path)
    config["path"] = line_path
    config["tree"] = line_tree
    return list_of_path_tree


def run_expe(config, agent_file="agents"):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = os.path.join(agent_file, f"agent_TD3_{timestamp}")

    os.makedirs(file_name, exist_ok=True)
    with open(os.path.join(file_name, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    current_py = os.path.abspath(__file__)
    shutil.copy(current_py, os.path.join(file_name, os.path.basename(current_py)))
    ## Path ##
    p_target = config["p_target"]
    p_0 = config["p_0"]
    path = config["path"]
    tree = config["tree"]
    print(f"Starting point : {p_0}")
    print(f"Target point : {path[-1]}")
    x = p_0

    ## Background flow ##
    uniform_bg = config["uniform_bg"]
    rankine_bg = config["rankine_bg"]
    both_rankine_and_uniform = config["uniform_bg"] and config["rankine_bg"]
    random_curve = config["random_curve"]
    u_bg = np.zeros(2)

    ## Environnement creation ##
    x_0 = config["x_0"]
    C = config["C"]
    steps_per_action = config["steps_per_action"]
    Dt_action = config["Dt_action"]
    Dt_sim = Dt_action / steps_per_action
    n_lookahead = config["n_lookahead"]
    env = MicroSwimmer(x_0, C, Dt_sim, config["velocity_bool"], n_lookahead,config['velocity_ahead'],config['add_action'])

    ## Environnement parameters ##
    t_max = config["t_max"]
    threshold = config["threshold"]
    beta = config["beta"]
    Dt_action = Dt_sim * steps_per_action
    D = config["D"]

    ## Agent and Replay Buffer ##
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3.TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    ## Training parameters ##
    nb_episode = config["nb_episode"]
    batch_size = config["batch_size"]
    eval_freq = config["eval_freq"]
    eval_episodes = config["eval_episodes"]
    episode_start_update = config["episode_start_update"]
    save_model = config["save_model"]
    episode_update = config["episode_per_update"]
    list_of_path_tree = None

    ## Creation of file ##
    save_path_result = f"./{file_name}/results"
    save_path_result_fig = os.path.join(save_path_result, "fig/")
    if not os.path.exists(save_path_result):
        os.makedirs(save_path_result)
        os.makedirs(save_path_result_fig)
    save_path_model = f"./{file_name}/models/agent"
    if save_model and not os.path.exists(save_path_model):
        os.makedirs(save_path_model)

    ## Load environment ##
    agent_to_load = config["load_model"]

    if agent_to_load != "":
        agent, config = load_agent_config(
            agent, eval_freq, random_curve, uniform_bg, rankine_bg
        )

    ## Training variables ##
    iter = 0
    episode_reward = 0
    episode_num = 1
    training_reward = []
    best_eval_result = -np.inf
    eval_list = []
    count_reach_target = 0

    ## Reset ##
    state, done = env.reset(tree, path), False

    ## Config for evaluation ##
    config_eval_bis = copy.deepcopy(config)
    config_eval_bis["uniform_bg"] = False
    config_eval_bis["rankine_bg"] = False
    if config["random_curve"]:
        list_of_path_tree = varying_curve_init(config)
        A = 2
        f = 4
        N = nb_episode / 2

        
    print("Both pertubations : ", config["uniform_bg"] and config["rankine_bg"])

    ########### TRAINING LOOP ###########
    while episode_num < nb_episode:
        iter += 1

        if iter % steps_per_action == 0 or iter == 1:
            action = agent.select_action(state)

        if episode_num > config["pertubation_after_episode"]:
            u_bg = velocity_func(x)

        next_state, reward, done, info = env.step(
            action, tree, path, p_target, beta, D, u_bg, threshold
        )
        x = info["x"]
        replay_buffer.add(state.flatten(), action, next_state.flatten(), reward, done)
        state = next_state
        episode_reward += reward

        ## Update ##
        if episode_num % episode_update == 0 and episode_num > episode_start_update:
            agent.train(replay_buffer, batch_size)

        ## Evolutive reward and success rate ##
        if done:
            count_reach_target += 1
            if agent_to_load != "":
                beta = beta * 1.0005

        ## Ending and Evaluation ##
        if done or iter * Dt_sim > t_max:
            
            training_reward.append(episode_reward)

            if (episode_num) % eval_freq == 0 and episode_num >= 10:
                print(
                    f"Total iter: {iter+1} Episode Num: {episode_num} Reward: {episode_reward:.3f} Success rate: {count_reach_target/eval_freq}"
                )
                print('----------------------------------------------------------------------------------------------------------------------------------------')
                path_save_fig = os.path.join(
                    save_path_result_fig, "training_reward.png"
                )
                eval_rew, _, _, _, _ = evaluate_agent(
                    agent=agent,
                    env=env,
                    eval_episodes=eval_episodes,
                    config=config_eval_bis,
                    save_path_result_fig=save_path_result_fig,
                    file_name="eval_during_training",
                    random_parameters=False,
                    list_of_path_tree=list_of_path_tree,
                    title="",
                    plot=True,
                    parameters=[],
                    plot_background=False,
                )
                visualize_streamline(
                    agent,
                    config_eval_bis,
                    f"eval_during_training_streamline_{beta:.3f}.png",
                    save_path_result_fig,
                    type="line",
                    title="",
                    k=0,
                    parameters=[],
                )
                eval_rew = mean(eval_rew)
                eval_list.append(eval_rew)
                print(f"Eval result : {eval_rew:.3f}")
                
                if best_eval_result < eval_rew and episode_num > (
                    config["pertubation_after_episode"] * 3
                ):
                    best_eval_result = eval_rew
                    if save_model:
                        agent.save(save_path_model)
                        print("Best reward during evaluation : Model saved")
                        
                episodes_values = np.linspace(1, episode_num, episode_num, dtype="int")
                episodes_values_freq = np.linspace(
                    10, episode_num, (episode_num - 10) // eval_freq + 1, dtype="int"
                )
                plt.plot(
                    episodes_values,
                    -np.array(training_reward),
                    color="blue",
                    label="training",
                )
                plt.plot(
                    episodes_values_freq,
                    -np.array(eval_list),
                    color="black",
                    label="evaluation",
                )
                plt.xlabel("episode")
                plt.ylabel("reward")
                plt.yscale("log")
                plt.legend()
                plt.savefig(path_save_fig, dpi=100, bbox_inches="tight")
                plt.close()
                if agent_to_load:
                    print("Beta increased : ", beta)

                count_reach_target = 0

            ## Reset and update parameters for next loop ##
            iter = 0
            episode_reward = 0
            episode_num += 1
            dir, norm, center, a, cir = random_bg_parameters()
            
            if both_rankine_and_uniform:
                velocity_func = lambda x: uniform_velocity(dir, norm) if episode_num % 2 == 0 else rankine_vortex(x, a, center, cir)
            elif uniform_bg:
                velocity_func = lambda x: uniform_velocity(dir, norm)
            elif rankine_bg:
                velocity_func = lambda x: rankine_vortex(x, a, center, cir)
            state, done = env.reset(tree, path,velocity_func), False

            if config["random_curve"] and episode_num > 10:
                k = func_k_max(A, N, f, episode_num)
                path = generate_curve_with_target_curvature(p_0, p_target, k, config["nb_points_path"])
                tree = KDTree(path)
                config["path"] = path
                config["tree"] = tree

    ## Save evaluation returns ##
    file_reward_eval = os.path.join(save_path_result, "rewards_eval")
    with open(file_reward_eval, "w") as f:
        json.dump(eval_list, f, indent=4)

def set_parameters_training(threshold,maximum_curv):
    l = 1 / maximum_curv
    Dt_action = 1 / maximum_curv
    D = threshold**2 / (20 * Dt_action)
    return Dt_action,D

if __name__ == "__main__":

    p_target = np.array([2, 0])
    p_0 = np.zeros(2)
    nb_points_path = 500
    path, d = generate_simple_line(p_0, p_target, nb_points_path)
    print(
        "Distance path points:      ",
        format_sci(np.linalg.norm(path[25, :] - path[0, :])),
    )
    tree = KDTree(path)
    print("Curvature max du chemin :  ", format_sci(np.max(courbures(path))))
    t_max = 8
    t_init = 0
    maximum_curvature = 30
    threshold=0.07
    Dt_action,D = set_parameters_training(threshold=threshold,maximum_curv=maximum_curvature)

    print("D:                         ", format_sci(D))
    print("Dt_action:                 ", format_sci(Dt_action))
    print("Threshold:                 ", format_sci(threshold))
    print("Mean diffusion distance:   ", format_sci(sqrt(2 * Dt_action * D)))
    print("Distance during Dt_action: ", format_sci(Dt_action))
    print("Distance to cover:         ", format_sci(d))
    print("Expected precision:        ", format_sci(threshold / d))
    config = {
        "x_0": p_0,  # m
        "C": 1,  # m/s
        "D": D,  # m2/s (Diffusion coefficient)
        "u_bg": np.array([0, 1]) * 0.0,  # m/s
        "threshold": threshold,  # m
        "t_max": t_max,  # s
        "t_init": t_init,  # s
        "steps_per_action": 5,
        "nb_episode": 650,
        "batch_size": 256,
        "eval_freq": 50,
        "save_model": True,
        "eval_episodes": 8,
        "episode_start_update": 10,
        "path": path,
        "tree": tree,
        "p_0": p_0,  # m
        "p_target": p_target,  # m
        "start_timesteps": 256,
        "load_model": "",
        "episode_per_update": 3,
        "discount_factor": 1,
        "beta": 0.25,
        "uniform_bg": True,  # Random uniform background flow during the training
        "rankine_bg": True,  # Random rankine vortex during the training
        "pertubation_after_episode": 20,  # Background flow add in the training after this episode
        "random_curve": True,  # To train on varying curve (no longer random)
        "nb_points_path": nb_points_path,  # Discretization of the path
        "Dt_action": Dt_action,
        "velocity_bool": False,  # Add the velocity in the state or not
        "n_lookahead": 10,  # Number of points in the lookahead
        "velocity_ahead":  False,
        'add_action' : False
    }

    run_expe(config)
