import os
import sys
import time
from math import sqrt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle
from math import ceil, cos, sin, sqrt

import numpy as np
from scipy.spatial import KDTree

from Astar import resample_path, shortcut_path

from .env_swimmer import MicroSwimmer
from .generate_path import *
from .TD3 import TD3

colors = plt.cm.tab10.colors
import copy
import random
from datetime import datetime
from itertools import chain
from pathlib import Path
from statistics import mean

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d

from .utils import create_numbered_run_folder
from .analytic_solution_line import find_next_v
from .Astar_ani import astar_anisotropic, compute_v, contour_2D
from .data_loader import load_sdf_from_csv, vel_read
from .distance_to_path import min_dist_closest_point
from .evaluate_agent import evaluate_agent
from .fmm import compute_fmm_path
from .invariant_state import coordinate_in_global_ref
from .plot import plot_action, plot_trajectories,plot_success_rate
from .rank_agents import rank_agents_by_rewards
from .sdf import get_contour_coordinates, sdf_circle, sdf_many_circle
from .simulation import solver
from .visualize import (
    plot_robust_D,
    plot_robust_u_bg_rankine,
    plot_robust_u_bg_uniform,
    visualize_streamline,
)


def format_sci(x):
    return "{:.3e}".format(x)


def evaluate_after_training(
    agent_files,
    obstacle_contour,
    seed=42,
    obstacle_type=None,
    velocity_func=None,
    plot_velocity_field=None,
    title_add="",
    list_config_paths=[],
    sigma = 10,
    description='',
):
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    results = {}

    uniform_bg = False
    rankine_bg = False


    file_path_result_global = "grid_search"
    file_path_result = str(create_numbered_run_folder(file_path_result_global))
    os.makedirs(file_path_result, exist_ok=True)

    file_name_result = os.path.join(
        file_path_result, f"result_evaluation_{obstacle_type}_{title_add}.json"
    )
    
    try:
        with open(file_name_result, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    for agent_name in agent_files:
        config_eval = initialize_parameters(agent_name, 0.00)
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

        print("Agent name : ", agent_name)
        config_eval["uniform_bg"] = uniform_bg
        config_eval["rankine_bg"] = rankine_bg
        config_eval["random_curve"] = False
        config_eval["beta"] = 0.25

        Dt_action = config_eval["Dt_action"]
        steps_per_action = config_eval["steps_per_action"]
        Dt_sim = Dt_action / steps_per_action
        threshold = config_eval["threshold"]
        success_rate_list = []

        results["type"] = [obstacle_type]

        for path_to_config in list_config_paths:
            if agent_name in results.keys():
                results_per_config = results[agent_name]["results_per_config"]
                if path_to_config in results[agent_name]["results_per_config"]:
                    continue
            else:
                results_per_config = {}
            (
                path,
                start_point,
                goal_point,
                sdf,
                velocity_retina,
                X_new,
                Y_new,
                time,
                ratio,
                parameters,
            ) = load_config_path(path_to_config)
            
            file_path_result_parameters = os.path.join(file_path_result,'parameters.json')
            parameters['sigma']=sigma
            parameters['description'] = description
            with open(file_path_result_parameters,'w') as f:
                json.dump(parameters,f,indent=4)
                
            smoothed_x = gaussian_filter1d(path[:, 0], sigma=15)
            smoothed_y = gaussian_filter1d(path[:, 1], sigma=15)
            path = np.stack([smoothed_x, smoothed_y], axis=1)
            tree = KDTree(path)
            config_eval["path"] = path
            config_eval["tree"] = tree
            config_eval["p_target"] = np.array(goal_point)
            config_eval["p_0"] = np.array(start_point)
            config_eval["x_0"] = np.array(start_point)
            print("Evaluation - Path ready")
            time_t = f"{time}"
            file_name_or = f"_{time_t}_obstacle_{obstacle_type}_{title_add}"

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
            save_path_eval = os.path.join(agent_name, f"eval_bg/velocity_ratio_{ratio}")

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
                f"eval_with" + file_name_or,
                False,
                title="",
                plot=True,
                plot_background=True,
                rng=rng,
                obstacle_contour=obstacle_contour,
                sdf=sdf,
                velocity_func=velocity_func,
            )

            plt.close()
            results_per_config[path_to_config] = {
                "rewards": rewards_per_episode,
                "rewards_time": rewards_t_per_episode,
                "rewards_distance": rewards_d_per_episode,
                "success_rate": success_rate,
            }
            success_rate_list.append(success_rate)

            results[agent_name] = {
                "mean_success_rate": mean(success_rate_list),
                "n_eval_episodes": config_eval["eval_episodes"],
                "training type": training_type,
                "results_per_config": results_per_config,
            }
            with open(file_name_result, "w") as f:
                json.dump(results, f, indent=4)
            plot_success_rate(file_name_result,agent_name,file_path_result)
        print("-----------------------------------------------")
        print("Success rate : ", mean(success_rate_list))
        print("-----------------------------------------------")

    threshold = [0.07]
    D = config_eval["D"]
    # plot_robust_D(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    config_eval["D"] = D
    # plot_robust_u_bg_uniform(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    # plot_robust_u_bg_rankine(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    plt.close()
    return results


def initialize_parameters(agent_file, bounce_thr):
    path_config = os.path.join(agent_file, "config.pkl")
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    config_eval = copy.deepcopy(config)
    config_eval["random_curve"] = (
        config["random_curve"] if "random_curve" in config else False
    )
    config_eval["nb_points_path"] = 500
    config_eval["t_max"] = 30
    config_eval["eval_episodes"] = 200
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
    threshold = 10 / 576 * 24
    config_eval["D"] = D
    config_eval["bounce_thr"] = bounce_thr
    return config_eval


def sdf_circle(point, center, radius):
    px, py = point
    cx, cy = center
    distance_to_center = sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return distance_to_center - radius


def sdf_func_and_velocity_func(domain_size, ratio):
    x, y, N, h, sdf = load_sdf_from_csv(domain_size)
    sdf_interpolator = RegularGridInterpolator(
        (y, x), sdf, bounds_error=False, fill_value=None
    )

    def sdf_function(point):
        return sdf_interpolator(point[::-1])

    path_vel = "data/vel.sdf"
    N, h, vel = vel_read(path_vel)
    v = vel[N[2] // 2, :, :, 0:2]
    vx, vy = v[:, :, 0], v[:, :, 1]
    velocity_interpolator_x = RegularGridInterpolator(
        (y, x), vx, bounds_error=False, fill_value=None
    )
    velocity_interpolator_y = RegularGridInterpolator(
        (y, x), vy, bounds_error=False, fill_value=None
    )
    v_magnitude = np.sqrt(vx**2 + vy**2)

    def velocity_retina(point):
        return (
            ratio
            * np.array(
                [
                    velocity_interpolator_x(point[::-1]),
                    velocity_interpolator_y(point[::-1]),
                ]
            )
            / np.max(v_magnitude)
        )

    return sdf_function, velocity_retina


def load_config_path(path_to_config_path):
    # If config_path_fmm is provided, other parameters can be None
    with open(path_to_config_path, "r") as f:
        config_path_fmm = json.load(f)

    parameters = config_path_fmm["parameters"]
    start_point = parameters["start_point"]
    goal_point = parameters["goal_point"]
    domain_size = parameters["domain_size"]
    grid_size = parameters["grid_size"]
    ratio = parameters["ratio"]
    B = parameters["B"]
    flow_factor = parameters["flow_factor"]
    time = parameters["current_time"]
    path = np.load(config_path_fmm["path_path"], allow_pickle=False)
    save_path_phi = config_path_fmm["path_phi"]
    save_path_flow = config_path_fmm["path_flow"]

    sdf_function, velocity_retina = sdf_func_and_velocity_func(domain_size, ratio)
    x_new = np.linspace(0, domain_size[0], parameters["grid_size"][0])
    y_new = np.linspace(0, domain_size[1], parameters["grid_size"][1])
    X_new, Y_new = np.meshgrid(x_new, y_new)
    return (
        path,
        start_point,
        goal_point,
        sdf_function,
        velocity_retina,
        X_new,
        Y_new,
        time,
        ratio,
        parameters
    )


def obstacle_and_path(
    scale=None,
    ratio=None,
    B=None,
    flow_factor=None,
    res_factor=None,
    start_point=None,
    goal_point=None,
    path_method=None,
    heuristic_weight=None,
    weight_sdf=None,
    path_to_config_path=None,
    ):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if path_to_config_path is not None:
        path, _, _, sdf_function, _, X_new, Y_new = load_config_path(
            path_to_config_path
        )
    else:
        # If config_path_fmm is None, all other parameters must be provided
        print("Creation of config file...")
        if None in [scale, ratio, B, flow_factor, res_factor, start_point, goal_point,weight_sdf,heuristic_weight]:
            raise ValueError(
                "If config_path_fmm is None, all other parameters (scale, ratio, B, flow_factor, res_factor, start_point, goal_point) must be provided."
            )

        config_path_fmm = {}

        domain_size = (1 * scale, 1 * scale)

        # Determine the size of the domain. It maps each point of the domain to each point on the grid.
        x, y, N, h, sdf = load_sdf_from_csv(domain_size)
        # Compute sdf and velocity interpolator regarding the size of the domain. x,y and sdf must have the same shape
        sdf_function, velocity_retina = sdf_func_and_velocity_func(domain_size, ratio)

        # sdf_function :  Calculate the sdf in any point of the domain py interpolation
        # velocity_retina :  Calculate the velocity in any point of the domain py interpolation

        start_point = (start_point[0] * scale, start_point[1] * scale)
        goal_point = (goal_point[0] * scale, goal_point[1] * scale)

        # Reduce the cell size by a factor : res_factor

        grid_size = (N[0] * res_factor, N[1] * res_factor)
        x_new = np.linspace(0, domain_size[0], grid_size[0])
        y_new = np.linspace(0, domain_size[1], grid_size[1])
        parameters = {
            "start_point": start_point,
            "goal_point": goal_point,
            "grid_size": grid_size,
            "domain_size": domain_size,
            "ratio": ratio,
            "B": B,
            "flow_factor": flow_factor,
            "heuristic_weight": heuristic_weight,
            "weight_sdf": weight_sdf,
            "method": path_method,
            "current_time": current_time,
        }
        config_path_fmm["parameters"] = parameters

        save_path_path = f"data/retina2D_path_time_{current_time}.npy"
        if path_method == "fmm":
            path, travel_time, grid_info, save_path_phi, save_path_flow = (
                compute_fmm_path(
                    start_point,
                    goal_point,
                    sdf_function,
                    x_new,
                    y_new,
                    B=B,
                    flow_field=velocity_retina,
                    grid_size=parameters["grid_size"],
                    domain_size=domain_size,
                    ratio=ratio,
                    flow_factor=flow_factor,
                )
            )
        elif path_method == "astar":
            # Compute v0,vx and vy on this new domain with a certain size of cell
            v0, vx, vy, save_path_phi, save_path_flow = compute_v(
                x_new, y_new, velocity_retina, B, grid_size, ratio, sdf_function
            )

            path, travel_time = astar_anisotropic(
                x_new,
                y_new,
                v0,
                vx,
                vy,
                start_point,
                goal_point,
                sdf_function,
                heuristic_weight=heuristic_weight,
            )

            path = np.array(path)  # de forme (N, 2)
            dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
            n = ceil(np.max(dist) / (5 * 1e-3))
            if n > 1:
                path = resample_path(path, len(path) * n)
        config_path_fmm["path_path"] = save_path_path
        config_path_fmm["path_phi"] = save_path_phi
        config_path_fmm["path_flow"] = save_path_flow

        np.save(save_path_path, path, allow_pickle=False)
        file_to_config_path = f"config_path/velocity_ratio_{ratio}"
        os.makedirs(file_to_config_path, exist_ok=True)
        path_to_config_path = os.path.join(
            file_to_config_path, f"config_path__{path_method}_{current_time}.json"
        )
        with open(path_to_config_path, "w") as f:
            json.dump(config_path_fmm, f, indent=4)
            print("Config saved ")

    X_new, Y_new = np.meshgrid(x_new, y_new)
    obstacle_contour = contour_2D(sdf_function, X_new, Y_new, scale)

    return (
        np.array(start_point),
        np.array(goal_point),
        sdf_function,
        path,
        obstacle_contour,
        velocity_retina,
        current_time,
    )


def create_list_of_goal_point(n, start_point):
    point_list = np.random.rand(n, 2)
    sdf_function, _ = sdf_func_and_velocity_func((1, 1), ratio)
    goal_points = [
        point
        for point in point_list
        if (
            sdf_function(point) < -0.3
            and point[0] < 0.9
            and np.linalg.norm(point - start_point) > 0.05
        )
    ]
    return goal_points


if __name__ == "__main__":
    obstacle_type = "retina"
    agents_file = []
    directory_path = Path("agents/")

    # for item in directory_path.iterdir():
    #     if item.is_dir() and "agent_TD3" in item.name:
    #         if "2025-04-23" in item.name or "2025-04-22" in item.name:
    #             agents_file.append(os.path.join(directory_path, item.name))
    agents_file = ["agents/agent_TD3_2025-04-18_13-33"]

    print("Agents files : ", agents_file)

    scale = 20
    ratio = 5
    N = (576, 528)
    start_point = (0.98, 0.3)
    res_factor = 1
    grid_size = (N[0] * res_factor, N[1] * res_factor)
    domain_size = (1 * scale, 1 * scale)
    x_new = np.linspace(0, domain_size[0], grid_size[0])
    y_new = np.linspace(0, domain_size[1], grid_size[1])
    X_new, Y_new = np.meshgrid(x_new, y_new)
    heuristic_weight = 0.1
    weight_sdf = 8
    sigma = 15
    description =''
    
    compute_goal_points = True

    # if compute_goal_points :
    #     goal_points = create_list_of_goal_point(50,start_point)
    #     for goal_point in goal_points:
    #         goal_point=tuple(goal_point)
    #         p_0, p_target, sdf_func, path, obstacle_contour, velocity_retina,current_time = obstacle_and_path(
    #             scale=scale,
    #             ratio=ratio,
    #             flow_factor=2,
    #             B=1,
    #             res_factor=res_factor,
    #             start_point=start_point,
    #             goal_point=goal_point,
    #             path_method='astar',
    #             heuristic_weight=heuristic_weight,
    #             weight_sdf = weight_sdf,
    #             path_to_config_path=None,
    #         )
    #     np.save('data/random_target_points',np.array(goal_points))
    # else :
    #     goal_points= np.load('data/random_target_points')
        
    start_time_eva = time.time()
    list_config_paths = []
    dir_config_path = Path(f"config_path/velocity_ratio_{ratio}/")

    for item in dir_config_path.rglob("*.json"):
        list_config_paths.append(os.path.join(dir_config_path, item.name))

    sdf_func, velocity_retina = sdf_func_and_velocity_func(domain_size, ratio)
    obstacle_contour = contour_2D(sdf_func, X_new, Y_new, scale)
    print("Number of different paths :",len(list_config_paths))
    print("Path generated - Go for evaluation")
    results = evaluate_after_training(
        agents_file,
        obstacle_contour=obstacle_contour,
        obstacle_type=obstacle_type,
        velocity_func=velocity_retina,
        list_config_paths=list_config_paths,
        sigma=15
    )
    end_time_eva = time.time()
    elapsed_time = (end_time_eva - start_time_eva) / 60
    print("Execution time:", elapsed_time, "minutes")
    # rank_agents_by_rewards(results)
