import os
import sys
from math import sqrt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle
from math import cos, sin, sqrt

import numpy as np
from scipy.spatial import KDTree

from Astar import resample_path, shortcut_path

from .env_swimmer import MicroSwimmer
from .generate_path import *
from .TD3 import TD3

colors = plt.cm.tab10.colors
import copy
import random
from pathlib import Path
from statistics import mean

from .analytic_solution_line import find_next_v
from .distance_to_path import min_dist_closest_point
from .evaluate_agent import evaluate_agent
from .find_shortest_path import find_shortest_path_circle
from .invariant_state import coordinate_in_global_ref
from .plot import plot_action, plot_trajectories
from .rank_agents import rank_agents_by_rewards
from .sdf import get_contour_coordinates, sdf_circle, sdf_many_circle
from .simulation import solver
from .visualize import (plot_robust_D, plot_robust_u_bg_rankine,
                        plot_robust_u_bg_uniform, visualize_streamline)
from .data_load import load_sdf_from_csv,vel_read
from .fmm import compute_fmm_path
from scipy.interpolate import RegularGridInterpolator

def format_sci(x):
    return "{:.3e}".format(x)


def evaluate_after_training(
    agent_files,
    p_target,
    p_0,
    path,
    obstacle_contour,
    sdf,
    seed=42,
    obstacle_type=None,
    velocity_func =None,
    plot_velocity_field=None,
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
    tree = KDTree(path)

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

    file_path_result = "results_evaluation/"
    file_name_or  =f"_{title_add}_obstacle_{obstacle_type}"
    os.makedirs(file_path_result, exist_ok=True)
    file_name_result = os.path.join(file_path_result, "result_evaluation"+file_name_or+".json")

    try:
        with open(file_name_result, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    results["type"] = [obstacle_type]

    for agent_name in agent_files:

        config_eval = initialize_parameters(agent_name, p_target, p_0, 0.01)
        config_eval = copy.deepcopy(config_eval)
        training_type = {
            "rankine_bg": config_eval["rankine_bg"],
            "uniform_bg": config_eval["uniform_bg"],
            "random_curve": config_eval["random_curve"],
            "velocity_bool": config_eval["velocity_bool"],
            "load_model": config_eval["load_model"],
            "n_lookahead": config_eval["n_lookahead"],
            "beta": config_eval["beta"],
        }
        # if agent_name in results.keys():
        #     results[agent_name]['training type'] = training_type
        #     print(f"Agent {agent_name} already evaluated.")
        #     continue
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
            config_eval["x_0"],
            config_eval["C"],
            Dt_sim,
            config_eval["velocity_bool"],
            config_eval["n_lookahead"],
            seed,
            config_eval["bounce_thr"],
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
            agent,
            env,
            config_eval["eval_episodes"],
            config_eval,
            save_path_eval,
            f"eval_with"+file_name_or,
            False,
            title="",
            plot=True,
            parameters=parameters,
            plot_background=True,
            rng=rng,
            obstacle_contour=obstacle_contour,
            sdf=sdf,
            velocity_func=velocity_func
        )

            
        plt.close()
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

    file_name_result = os.path.join(file_path_result, file_name_or)
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


def compare_p_line(
    agent_name,
    config_eval,
    file_name_or,
    save_path_eval,
    u_bg=np.zeros(2),
    title="",
    offset=0.2,
):
    config_eval = copy.deepcopy(config_eval)
    save_path_comparison = os.path.join(save_path_eval, "comparison_graph/")
    if not os.path.exists(save_path_comparison):
        os.makedirs(save_path_comparison)
    trajectories = {}
    p_target = config_eval["p_target"]
    p_0 = config_eval["p_0"]
    nb_points_path = config_eval["nb_points_path"]
    nb_starting_point = 4
    p_0_above = p_0 + np.array([0.2, offset])
    p_target_above = p_target + np.array([0, offset])
    p_0_below = p_0 + np.array([0.2, -offset])
    p_target_below = p_target + np.array([0, -offset])

    path, _ = generate_simple_line(p_0, p_target, nb_points_path)
    path_above_point, _ = generate_simple_line(
        p_0_above, p_target_above, nb_starting_point
    )
    path_below_point, _ = generate_simple_line(
        p_0_below, p_target_below, nb_starting_point
    )

    config_eval["path"] = path
    tree = KDTree(path)
    config_eval["tree"] = tree

    path_above_point = path_above_point[:-1]
    path_below_point = path_below_point[:-1]
    path_starting_point = np.concatenate((path_above_point, path_below_point), axis=0)
    L = len(path_starting_point)
    print("path_starting_point : ", path_starting_point)
    file_name_or += title
    path_save_fig = os.path.join(save_path_comparison, file_name_or)

    config_eval["D"] = 0
    steps_per_action = config_eval["steps_per_action"] - 2
    t_max = config_eval["t_max"]
    Dt_action = config_eval["Dt_action"]
    Dt_sim = Dt_action / steps_per_action

    id = 0
    starting_point = path_starting_point[id]
    dot_product = []
    config_eval["x_0"] = starting_point
    x = starting_point
    env = MicroSwimmer(
        config_eval["x_0"],
        config_eval["C"],
        Dt_sim,
        config_eval["velocity_bool"],
        config_eval["n_lookahead"],
    )
    state, done = env.reset(config_eval["tree"], config_eval["path"]), False
    iter = 0

    policy_file = os.path.join(agent_name, "models/agent")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3.TD3(state_dim, action_dim, max_action)
    agent.load(policy_file)
    states_list = [x]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(
        path[:, 0], path[:, 1], label="path", color="black", linewidth=2, zorder=10
    )

    while id < len(path_starting_point):
        iter += 1
        if iter % steps_per_action == 0 or iter == 1:
            action = agent.select_action(state)
            v, p_dir = find_next_v(
                x, config_eval["p_target"], config_eval["beta"], Dt_sim, num_angles=90
            )
            x_previous = x
        next_state, reward, done, info = env.step(
            action,
            tree,
            path,
            p_target,
            config_eval["beta"],
            config_eval["D"],
            u_bg,
            config_eval["threshold"],
        )
        x = info["x"]
        states_list.append(x)
        state = next_state

        if iter % steps_per_action == 1:
            dir_act = (x - x_previous) / np.linalg.norm(x - x_previous)
            product = dir_act @ p_dir
            dot_product.append(product)

        if done or iter * Dt_sim > t_max:
            trajectories[f"{path_starting_point[id]}"] = dot_product
            color = plot_trajectories(
                ax1, np.array(states_list), path, title="streamlines", color_id=id
            )

            p_1 = path_starting_point[id]
            x_values = np.linspace(p_1[0], p_target[0], len(dot_product))
            ax2.plot(x_values, dot_product, color=color, label=f"Traj {id}")

            iter = 0
            id += 1
            if id < L:
                x_0 = path_starting_point[id]
                config_eval["x_0"] = x_0
                dot_product = []
                states_list = [x_0]
                x = x_0
                env = MicroSwimmer(
                    config_eval["x_0"],
                    config_eval["C"],
                    Dt_sim,
                    config_eval["velocity_bool"],
                    config_eval["n_lookahead"],
                )
                state, done = env.reset(config_eval["tree"], config_eval["path"]), False
            if ax1.get_ylim()[1] - ax1.get_ylim()[0] < 0.5:
                ax1.set_ylim(-0.3, 0.3)

    ax1.set_aspect("equal")
    ax1.set_title("Trajectories")
    ax1.legend()

    ax2.set_aspect("equal")
    ax2.set_title("Dot Product Evolution")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path_save_fig, dpi=200, bbox_inches="tight")


def policy_direction(agent_name, config_eval):
    p_0 = np.zeros(2)
    p_target = np.array([1 / 8, 1 / 64])
    k = -0.4
    nb_points = 100
    path = generate_curve(p_0, p_target, k, nb_points)
    tree = KDTree(path)
    x_0 = np.array([0.02, -0.01])
    x = x_0
    config_eval["x_0"] = x_0
    env = MicroSwimmer(
        config_eval["x_0"],
        config_eval["C"],
        config_eval["Dt_action"] / config_eval["steps_per_action"],
    )
    beta = config_eval["beta"]
    u_bg = np.zeros(2)
    D = config_eval["D"]
    threshold = config_eval["threshold"]

    policy_file = os.path.join(agent_name, "models/agent")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3.TD3(state_dim, action_dim, max_action)
    agent.load(policy_file)

    nb_steps = 5
    state, done = env.reset(tree, path), False

    plt.plot(path[:, 0], path[:, 1], label="path", color="black", linewidth=2)
    plt.scatter(p_0[0], p_0[1], color="black")
    for i in range(nb_steps):
        action = agent.select_action(state)
        d, id_cp = min_dist_closest_point(x, tree)
        dir_path = path[id_cp + 1] - path[id_cp]
        action_global = coordinate_in_global_ref(path[id_cp], dir_path, action)
        plot_action(path, x, p_0, id_cp, action_global, i)
        next_state, x, reward, done, _ = env.step(
            action, tree, path, p_target, beta, D, u_bg, threshold
        )
        state = next_state
    save_path_eval_action = os.path.join(agent_name, "eval_bg/action_choice")

    plt.legend()
    plt.savefig(save_path_eval_action, dpi=200, bbox_inches="tight")


def initialize_parameters(agent_file, p_target, p_0, bounce_thr):
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
    config_eval["nb_points_path"] = 500
    config_eval["t_max"] = 20
    config_eval["eval_episodes"] = 100
    config_eval["velocity_bool"] = (
        config["velocity_bool"] if "velocity_bool" in config else False
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
    config_eval["bounce_thr"] = bounce_thr
    return config_eval


def sdf_circle(point, center, radius):
    px, py = point
    cx, cy = center
    distance_to_center = sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return distance_to_center - radius


def obstacle_and_path_a_star(obstacle_type):
    if obstacle_type == "circle":

        def sdf(point):
            centers = [(-1, 1 / 2), (1, -1 / 2)]
            radius = 3 / 4
            return sdf_many_circle(point, centers, radius)

        x = np.linspace(-2, 2, 500)
        y = np.linspace(-2, 2, 500)
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.vectorize(lambda px, py: sdf((px, py)))(X, Y)

        start = [0, 250]
        goal = [490, 250]
        obstacle_contour = get_contour_coordinates(X, Y, Z, level=0)

    p_0 = np.array([x[start[0]], y[start[1]]])
    p_target = np.array([x[goal[0]], y[goal[1]]])

    maximum_curvature = 30
    Dt_action = 1 / maximum_curvature
    threshold = 0.07
    D = threshold**2 / (20 * Dt_action)
    min_distance = 2 * sqrt(2 * Dt_action * D)

    path_indices = find_shortest_path_circle(start, goal, min_distance, Z)
    path_indices_short = shortcut_path(path_indices, X, Y, Z, min_distance)
    path_coords_short = np.array(
        [[X[idx[0], idx[1]], Y[idx[0], idx[1]]] for idx in path_indices_short]
    )
    path = resample_path(path_coords_short, n_points=500)
    return p_0, p_target, sdf, path, obstacle_contour

def obstacle_and_path_fmm(obstacle_type):
    scale = 8
    domain_size =(1*scale,1*scale)
 
    x,y,N,h,sdf = load_sdf_from_csv(domain_size)
    grid_size = (N[0],N[1])
    # np.save(f"data/sdf_retina2D_scale_{scale}.npy", sdf, allow_pickle=False)
    X, Y = np.meshgrid(x, y)

    start_point = (0.98*scale, 0.3*scale)
    goal_point = (0.44*scale, 0.55*scale)
    sdf_interpolator = RegularGridInterpolator((y, x), sdf, bounds_error=False, fill_value=None)
    
    def sdf_function(point):
        return sdf_interpolator(point[::-1]) 
            
    path_vel = 'data/vel.sdf'
    N, h, vel = vel_read(path_vel)
    v = vel[N[2]//2,:,:,0:2]
    vx = v[:,:,0]
    vy = v[:,:,1]
    velocity_interpolator_x = RegularGridInterpolator((y, x), vx, bounds_error=False, fill_value=None)
    velocity_interpolator_y = RegularGridInterpolator((y, x), vy, bounds_error=False, fill_value=None)
    v = np.sqrt(vx**2 + vy**2)
    def velocity_retina(point):
        return 2.5*np.array([velocity_interpolator_x(point[::-1]),velocity_interpolator_y(point[::-1])])/(np.max(v))
 
        
    if os.path.exists(f"data/retina2D_path_scale_{scale}.npy"):
        path = np.load(f"data/retina2D_path_scale_{scale}.npy", allow_pickle=False)
        print("Path loaded")
    else : 
        path, travel_time, grid_info = compute_fmm_path(start_point, goal_point, sdf_function, x, y, B=20,flow_field=velocity_retina,grid_size=grid_size,domain_size=domain_size)
        path=np.array(path)
        np.save(f"data/retina2D_path_scale_{scale}.npy", path, allow_pickle=False)
    if os.path.exists(f"data/retina2D_contour_scale{scale}.npy"):
        obstacle_contour = np.load(f"data/retina2D_contour_scale{scale}.npy", allow_pickle=False)
        print("Contour loaded")
    else : 
        Z = np.vectorize(lambda px, py: sdf_function((px, py)))(X, Y)
        obstacle_contour = get_contour_coordinates(X, Y, Z, level=0)
        np.save(f"data/retina2D_contour_scale{scale}.npy",obstacle_contour,allow_pickle=False)
        

    
    return np.array(start_point),np.array(goal_point),sdf_function,path,obstacle_contour,velocity_retina

if __name__ == "__main__":
    obstacle_type = "retina"
    agents_file = []
    directory_path = Path("agents/")

    for item in directory_path.iterdir():
        if item.is_dir() and "agent_TD3" in item.name:
            if "2025-04-23" in item.name or "2025-04-22" in item.name:
                agents_file.append(os.path.join(directory_path, item.name))
    agents_file = ["agents/agent_TD3_2025-04-18_13-33"]
    print("Agents files : ", agents_file)

    print("---------------------Evaluation with no bg---------------------")
    title_add = 'velocity_25_'
    p_0, p_target, sdf, path, obstacle_contour,velocity_retina = obstacle_and_path_fmm(obstacle_type)
    path=np.array(path)
    print("Path generated")
    results = evaluate_after_training(
        agents_file,
        p_target=p_target,
        p_0=p_0,
        path=path,
        obstacle_contour=obstacle_contour,
        sdf=sdf,
        title_add=title_add,
        obstacle_type=obstacle_type,
        velocity_func = velocity_retina,
    )
    rank_agents_by_rewards(results)

    # norm = 0.5
    # dict = {
    #     "east_05": np.array([1, 0]),
    #     "west_05": np.array([-1, 0]),
    #     "north_05": np.array([0, 1]),
    #     "south_05": np.array([0, -1]),
    # }
    # print("---------------------Evaluation with uniform bg---------------------")
    # for title_add, dir in dict.items():
    #     results = evaluate_after_training(
    #         agents_file,
    #         p_target=p_target,
    #         p_0=p_0,
    #         path=path,
    #         obstacle_contour=obstacle_contour,
    #         sdf=sdf,
    #         obstacle_type=obstacle_type,
    #         title_add=title_add,
    #         dir=dir,
    #         norm=norm,
    #     )
    #     rank_agents_by_rewards(results)

    # title_add = 'rankine_a_05__cir_3_center_1_075'
    # print("---------------------Evaluation with rankine bg---------------------")
    # a= 0.5
    # cir = 2
    # center = np.array([1,3/4])
    # results = evaluate_after_training(agents_file,f'result_evaluation_{title_add}_{type}.json',p_target = p_target,p_0 = p_0,path=path,obstacle_contour=obstacle_contour,title_add=title_add,a=a,center=center,cir=cir)
