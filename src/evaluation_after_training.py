import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle
from math import cos, sin

import numpy as np
from scipy.spatial import KDTree

from src.env_swimmer import MicroSwimmer
from src.generate_path import *
from src.TD3 import TD3

colors = plt.cm.tab10.colors
import copy
import random
from pathlib import Path
from statistics import mean

from src.analytic_solution_line import find_next_v
from src.distance_to_path import min_dist_closest_point
from src.evaluate_agent import evaluate_agent
from src.invariant_state import coordinate_in_global_ref
from src.plot import plot_action, plot_trajectories
from src.rank_agents import rank_agents_by_rewards
from src.simulation import solver
from src.visualize import (
    plot_robust_D,
    plot_robust_u_bg_rankine,
    plot_robust_u_bg_uniform,
    visualize_streamline,
)


def format_sci(x):
    return "{:.3e}".format(x)


def evaluate_after_training(
    agent_files,
    seed=42,
    type=None,
    translation=None,
    theta=None,
    dir=None,
    norm=None,
    a=None,
    center=None,
    cir=None,
    title_add="",
):

    nb_points_path = 2000
    radius = 1/2
    pitch = 1
    turns=1
    k = 0
    if type=='helix':
        path = generate_helix(nb_points_path, radius, pitch,turns,False)
    tree = KDTree(path)
    p_0 = path[0]
    p_target = path[-1]
    results = {}

    uniform_bg = False
    rankine_bg = False
    if dir is not None and norm is not None:
        dir = np.array(dir)
        norm = norm
        parameters = [dir, norm]
        uniform_bg = True
    elif a is not None and center is not None and cir is not None:
        parameters = [center, a, cir]
        rankine_bg = True
    else:
        parameters = []

    file_name_or = f"_{title_add}_{type}"
    file_path_result = "results_evaluation/"
    os.makedirs(file_path_result, exist_ok=True)
    file_name_result = os.path.join(
        file_path_result, "result_evaluation" + file_name_or + ".json"
    )

    try:
        with open(file_name_result, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    results["type"] = [type]

    for agent_name in agent_files:

        config_eval = initialize_parameters(agent_name, p_target, p_0)
        config_eval = copy.deepcopy(config_eval)
        training_type = {
            "rankine_bg": config_eval["rankine_bg"],
            "uniform_bg": config_eval["uniform_bg"],
            "random_curve": config_eval["random_curve"],
            "velocity_bool": config_eval["velocity_bool"],
            "load_model": config_eval["load_model"],
            "n_lookahead": config_eval["n_lookahead"],
            "beta": config_eval["beta"],
            "add_action": config_eval["add_action"],
            "velocity_ahead": config_eval["velocity_ahead"],
        }
        if agent_name in results.keys():
            results[agent_name]["training type"] = training_type
            print(f"Agent {agent_name} already evaluated.")
            continue
        print("Agent name : ", agent_name)
        config_eval["uniform_bg"] = uniform_bg
        config_eval["rankine_bg"] = rankine_bg
        config_eval["random_curve"] = False
        config_eval["beta"] = 0.25

        Dt_action = config_eval["Dt_action"]
        steps_per_action = config_eval["steps_per_action"]
        Dt_sim = Dt_action / steps_per_action
        threshold = config_eval["threshold"]

        config_eval["path"] = path
        config_eval["tree"] = tree
        env = MicroSwimmer(
            x_0 = config_eval["x_0"],
            C = config_eval["C"],
            Dt = Dt_sim,
            velocity_bool = config_eval["velocity_bool"],
            n_lookahead = config_eval["n_lookahead"],
            velocity_ahead= config_eval["velocity_ahead"],
            add_action = config_eval["add_action"],
            dim = config_eval["dim"],
        )

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = TD3(state_dim, action_dim, max_action)
        save_path_eval = os.path.join(agent_name, "eval_bg/")

        os.makedirs(save_path_eval, exist_ok=True)

        policy_file = os.path.join(agent_name, "models/agent")
        agent.load(policy_file)
        (
            rewards_per_episode,
            rewards_t_per_episode,
            rewards_d_per_episode,
            success_rate,
            _,
        ) = evaluate_agent(
            agent=agent,
            env=env,
            eval_episodes=config_eval["eval_episodes"],
            config=config_eval,
            save_path_result_fig=save_path_eval,
            file_name=f"eval_with" + file_name_or,
            random_parameters=False,
            title="",
            plot=True,
            parameters=parameters,
            plot_background=True,
        )

        results[agent_name] = {
            "rewards": rewards_per_episode,
            "rewards_time": rewards_t_per_episode,
            "rewards_distance": rewards_d_per_episode,
            "success_rate": success_rate,
            "n_eval_episodes": config_eval["eval_episodes"],
            "training type": training_type,
        }
        print("-----------------------------------------------")
        print("Success rate : ", success_rate)
        print("Mean rewards : ", format_sci(mean(rewards_per_episode)))
        print("Mean rewards t : ", format_sci(mean(rewards_t_per_episode)))
        print("Mean rewards d : ", format_sci(mean(rewards_d_per_episode)))
        print("-----------------------------------------------")


    with open(file_name_result, "w") as f:
        json.dump(results, f, indent=4)

    threshold = [0.07]
    D = config_eval["D"]
    # plot_robust_D(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    config_eval["D"] = D
    # plot_robust_u_bg_uniform(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    # plot_robust_u_bg_rankine(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    plt.close()
    return results





def initialize_parameters(agent_file, p_target, p_0):
    path_config = os.path.join(agent_file, "config.pkl")
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    config_eval = copy.deepcopy(config)
    config_eval["random_curve"] = (
        config["random_curve"] if "random_curve" in config else False
    )
    config_eval["p_target"] = p_target
    config_eval["p_0"] = p_0
    config_eval["x_0"] = p_0
    config_eval["nb_points_path"] = 2000
    config_eval["t_max"] = 30
    config_eval["eval_episodes"] = 10
    config_eval["velocity_bool"] = (
        config["velocity_bool"] if "velocity_bool" in config else False
    )
    config_eval["velocity_ahead"] = (
        config["velocity_ahead"] if "velocity_ahead" in config else False
    )
    config_eval["add_action"] = (
        config["add_action"] if "add_action" in config else False
    )
    config_eval["Dt_action"] = (
        config_eval["Dt_action"] if "Dt_action" in config else 1 / 30
    )
    config_eval["n_lookahead"] = config["n_lookahead"] if "n_lookahead" in config else 5
    maximum_curvature = 30
    l = 1 / maximum_curvature
    Dt_action = 1 / maximum_curvature
    threshold = 0.07
    D = threshold**2 / (20 * Dt_action)
    config_eval["D"] = D
    return config_eval


if __name__ == "__main__":
    agent_name = "agents/agent_TD3_2025-05-16_14-56"
    # p_target = np.array([2,0])
    # p_0 = np.array([0,0])
    # u_bg = np.array([0.0,0.2])
    # config_eval_comp = initialize_parameters(agent_name,p_target,p_0)
    # save_path_eval = os.path.join(agent_name,'eval_bg/')
    # os.makedirs(save_path_eval, exist_ok=True)
    # offset=0.2
    # compare_p_line(agent_name,config_eval_comp,'comparison_north_02',save_path_eval,u_bg,'',offset)

    print("Agents files : ", agents_file)
    types = ["ondulating", "line"]
    title_add = "rankine_a_05__cir_3_center_1_075"
    print("--------------------- Evaluation with rankine bg ---------------------")
    for type in types:
        a = 0.5
        cir = 2
        center = np.array([1, 3 / 4])
        results = evaluate_after_training(
            agents_file,
            type=type,
            p_target=[2, 0],
            p_0=[0, 0],
            title_add=title_add,
            a=a,
            center=center,
            cir=cir,
        )
        rank_agents_by_rewards(results)

    print("--------------------- Evaluation with no bg ---------------------")
    title_add = "free"
    for type in types:
        results = evaluate_after_training(
            agents_file,
            type=type,
            p_target=[2, 0],
            p_0=[0, 0],
            title_add=title_add,
        )
        rank_agents_by_rewards(results)

    norm = 0.5
    dict = {
        "east_05": np.array([1, 0]),
        "west_05": np.array([-1, 0]),
        "north_05": np.array([0, 1]),
        "south_05": np.array([0, -1]),
    }
    print("---------------------Evaluation with uniform bg---------------------")
    for type in types:
        for title_add, dir in dict.items():
            results = evaluate_after_training(
                agents_file,
                type=type,
                p_target=[2, 0],
                p_0=[0, 0],
                title_add=title_add,
                dir=dir,
                norm=norm,
            )
            rank_agents_by_rewards(results)



