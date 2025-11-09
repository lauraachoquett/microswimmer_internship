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
from itertools import chain

from src.performance_agent import performance_stats_plot,box_plot_data_free,box_plot_data
from src.evaluate_agent import evaluate_agent
from src.rank_agents import rank_agents_by_rewards
from src.generate_path import length_path


def format_sci(x):
    return "{:.3e}".format(x)


def evaluate_after_training(
    agent_files,
    p_target,
    p_0,
    seed=42,
    ref=False,
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
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    if translation is not None:
        translation = translation
    else:
        translation = np.zeros(2)

    if theta is not None:
        theta = theta
        sin_th = sin(theta)
        cos_th = cos(theta)
        R = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    else:
        R = np.eye(2)

    p_target = R @ p_target + translation
    p_0 = R @ p_0 + translation
    p_1 = [1 / 4, -1 / 4] + translation
    nb_points_path = 500
    k = 0
    if type == "line":
        path, _ = generate_simple_line(p_0, p_target, nb_points_path)
    if type == "two_line":
        path, _ = generate_line_two_part(p_0, p_1, p_target, nb_points_path)
    if type == "circle":
        path, _ = generate_demi_circle_path(p_0, p_target, nb_points_path)
    if type == "ondulating":
        path = generate_random_ondulating_path_old(
            p_0, p_target, n_points=700, amplitude=0.5, frequency=2
        )
    if type == "ondulating_hard":
        path = generate_random_ondulating_path(
            p_0, p_target, n_points=700, kappa_max=17, frequency=2
        )
    if type == "curve_minus":
        k = -3.43
        path = generate_curve_with_target_curvature(p_0, p_target, k, nb_points_path)
    if type == "curve_plus":
        k = 3.43
        path = generate_curve_with_target_curvature(p_0, p_target, k, nb_points_path)
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
        if ref :
            config_eval['D']=0
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
            # continue
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
            plot_t_l_d_per_episode
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
            "rewards_mean":mean(rewards_per_episode),
            "rewards_time": rewards_t_per_episode,
            "rewards_time_mean": mean(rewards_t_per_episode),
            "rewards_distance": rewards_d_per_episode,
            "rewards_distance_mean": mean(rewards_d_per_episode),
            "success_rate": success_rate,
            "n_eval_episodes": config_eval["eval_episodes"],
            "training type": training_type,
            "plot_t_l_d":plot_t_l_d_per_episode,
            "length_path":np.sum(length_path(path) )
            
        }
        print("-----------------------------------------------")
        # print("Success rate : ", success_rate)
        print("Mean rewards : ", format_sci(mean(rewards_per_episode)))
        # print("Mean rewards t : ", format_sci(mean(rewards_t_per_episode)))
        # print("Mean rewards d : ", format_sci(mean(rewards_d_per_episode)))
        # print("-----------------------------------------------")


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
    config_eval["eval_episodes"] = 50
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

    
    # agents_file=[agent_name,
    #              "agents/agent_TD3_2025-07-24_10-12",
    #              "agents/agent_TD3_2025-07-24_16-45",
    #              "agents/agent_TD3_2025-07-24_11-02",
    #              "agents/agent_TD3_2025-07-24_11-49",
    #              "agents/agent_TD3_2025-07-24_12-13",
    #              "agents/agent_TD3_2025-07-24_12-38",
    #              "agents/agent_TD3_2025-07-24_13-03",
    #              "agents/agent_TD3_2025-07-24_13-27",
    #              "agents/agent_TD3_2025-07-24_13-57",
                 
    #              ]
    agents_file = ['agents/agent_TD3_2025-05-07_15-48']

    # with open('data/list_n_lookahaed_08-12.json') as f:
    #     d = json.load(f)
    #     agents_file=list(chain.from_iterable(d.values()))
           
    # agent_name = "agents/agent_TD3_2025-04-18_13-33" 
    # agents_file = [agent_name,"agents/agent_TD3_2025-05-07_15-48"]
    
    print("Agents files : ", agents_file)
    # types = ["ondulating","line","curve_minus","curve_plus"]
    # labels=['S-curve','Straight Line','Convex Curve','Concave Curve']
    types = ["ondulating","line"]
    labels=['S-curve','Straight Line']
    rankine=False
    uniform=True
    free = False

    print("--------------------- Evaluation with no bg ---------------------")
    title_add = "free_D_0"
    for type in types:
        results = evaluate_after_training(
            ["agents/agent_TD3_2025-04-18_13-33"],
            type=type,
            p_target=[2, 0],
            p_0=[0, 0],
            title_add=title_add,
            ref=True
        )
        rank_agents_by_rewards(results)
        
    if rankine : 
        title_add = "rankine_a_05_cir_0_75_2_pi_center_1_0"
        print("--------------------- Evaluation with rankine bg ---------------------")
        for id,type in enumerate(types):
            a = 0.5
            umax =  (3./4.)
            cir = umax * (2. * np.pi * a)
            
            center = np.array([1,0])
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
            for agent_name in agents_file:
                file_path = f"results_evaluation/result_evaluation_{title_add}_{type}.json"
                file_path_free = f"results_evaluation/result_evaluation_free_D_0_{type}.json"
                performance_stats_plot(file_path=file_path,file_path_free=file_path_free,agent_name=agent_name,type=title_add,path=type,label = labels[id])

    if free : 
        print("--------------------- Evaluation with no bg ---------------------")
        title_add = "free"
        for id,type in enumerate(types):
            results = evaluate_after_training(
                agents_file,
                type=type,
                p_target=[2, 0],
                p_0=[0, 0],
                title_add=title_add,
            )
            rank_agents_by_rewards(results)
            for agent_name in agents_file:
                    file_path = f"results_evaluation/result_evaluation_{title_add}_{type}.json"
                    file_path_free = f"results_evaluation/result_evaluation_free_D_0_{type}.json"
                    performance_stats_plot(file_path=file_path,file_path_free=file_path_free,agent_name=agent_name,type=title_add,path=type,label = labels[id])

    if uniform:
        norm_list = np.linspace(0.5,1.1,20)
        dict = {
            "east": np.array([1, 0]),
            "west": np.array([-1, 0]),
            "north": np.array([0, 1]),
            "south": np.array([0, -1]),
        }
        dict_normed = {}
        for name, vec in dict.items():
            for norm in norm_list:
                # on arrondit à deux décimales et on formate sans le point
                key = f"{name}_{norm:.2f}".replace(".", "")
                dict_normed[key] = {
                    "vec": vec * norm,
                    "norm": norm
                }


        print("---------------------Evaluation with uniform bg---------------------")
        for id,type in enumerate(types):
            for title_add, value in dict_normed.items():
                dir =value['vec']
                norm =value['norm']
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
                # for agent_name in agents_file:
                #     file_path = f"results_evaluation/result_evaluation_{title_add}_{type}.json"
                #     file_path_free = f"results_evaluation/result_evaluation_free_D_0_{type}.json"
                #     performance_stats_plot(file_path=file_path,file_path_free=file_path_free,agent_name=agent_name,type=title_add,path=type,label = labels[id])
                    
                    
    # for agent_name in agents_file:
    #     summary_path =  os.path.join(agent_name,"performance_summary.json")
    #     box_plot_data_free(summary_path,agent_name)
    #     box_plot_data(summary_path, agent_name)

