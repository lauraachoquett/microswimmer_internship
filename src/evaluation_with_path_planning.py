import os
import sys
import time
from math import sqrt
import yaml

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle
from math import ceil, cos, sin, sqrt

import numpy as np
from scipy.spatial import KDTree

from Astar import resample_path, shortcut_path
from src.env_swimmer import MicroSwimmer
from src.generate_path import *
from src.TD3 import TD3

colors = plt.cm.tab10.colors
import copy
import os
import random
import sys
from datetime import datetime
from itertools import chain
from pathlib import Path
from statistics import mean

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d

from src.analytic_solution_line import find_next_v
from src.Astar_ani import astar_anisotropic, compute_v, contour_2D
from src.data_loader import load_sdf_from_csv, vel_read, load_sim_sdf
from src.distance_to_path import min_dist_closest_point
from src.evaluate_agent import evaluate_agent
from src.fmm import compute_fmm_path
from src.invariant_state import coordinate_in_global_ref
from src.plot import plot_action, plot_success_rate, plot_trajectories
from src.rank_agents import rank_agents_by_rewards
from src.sdf import get_contour_coordinates, sdf_circle, sdf_many_circle
from src.simulation import solver
from src.utils import create_numbered_run_folder
from src.visualize import (
    plot_robust_D,
    plot_robust_u_bg_rankine,
    plot_robust_u_bg_uniform,
    visualize_streamline,
)

# Ajouter le dossier 'src' au sys.path pour permettre l'importation des modules dans src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


def format_sci(x):
    return "{:.3e}".format(x)


def evaluate_after_training(
    agent_files,
    obstacle_contour,
    seed=42,
    obstacle_type=None,
    velocity_func=None,
    sdf_func=None,
    plot_velocity_field=None,
    title_add="",
    list_config_paths=[],
    sigma=10,
    file_path_result=None,
    ):
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    results = {}

    uniform_bg = False
    rankine_bg = False
    if file_path_result is None :
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
            "add_action": config_eval["add_action"],
            "velocity_ahead": config_eval["velocity_ahead"],
        }
        print("Training type :", training_type)
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

        for path_to_config in tqdm(list_config_paths, desc="Processing configs"):
            if agent_name in results.keys():
                results_per_config = results[agent_name]["results_per_config"]
                if path_to_config in results[agent_name]["results_per_config"]:
                    print('already evaluated')
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
            print("Path length  : ",len(path))
            if len(path) == 0 :
                print("Path empty : ", path_config['path_path'])
                continue
            file_path_result_parameters = os.path.join(
                file_path_result, "parameters.json"
            )
            parameters["sigma"] = sigma
            with open(file_path_result_parameters, "w") as f:
                json.dump(parameters, f, indent=4)

            smoothed_x = gaussian_filter1d(path[:, 0], sigma=15)
            smoothed_y = gaussian_filter1d(path[:, 1], sigma=15)
            path = np.stack([smoothed_x, smoothed_y], axis=1)
            tree = KDTree(path)
            config_eval["path"] = path
            config_eval["tree"] = tree
            config_eval["p_target"] = np.array(goal_point)
            config_eval["p_0"] = np.array(start_point)
            config_eval["x_0"] = np.array(start_point)
            time_t = f"{time}"
            file_name_or = f"_{time_t}_obstacle_{obstacle_type}_{title_add}"

            env = MicroSwimmer(
                config_eval["x_0"],
                config_eval["C"],
                Dt_sim,
                config_eval["velocity_bool"],
                config_eval["n_lookahead"],
                config_eval["velocity_ahead"],
                config_eval["add_action"],
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
                states_per_epsiode,
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
                sdf=sdf_func,
                velocity_func_l=velocity_func,
                video=False
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
            plot_success_rate(file_name_result, agent_name, file_path_result)
        print("-----------------------------------------------")
        print("Success rate : ", mean(success_rate_list))
        print("-----------------------------------------------")

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
    config_eval["eval_episodes"] = 100
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
    threshold = 5 / 576 * 24
    config_eval["threshold"] = threshold
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
    with open(path_to_config_path, "r") as f:
        config_path_a = json.load(f)

    parameters = config_path_a["parameters"]
    start_point = parameters["start_point"]
    goal_point = parameters["goal_point"]
    grid_size = parameters["grid_size"]
    ratio = parameters["ratio"]
    time = parameters["current_time"]
    path = np.load(config_path_a["path_path"], allow_pickle=False)
    sdf_func,velocity_retina,x_phys,y_phys,physical_width,physical_height,scale= load_sim_sdf(ratio)
    X,Y = np.meshgrid(x_phys,y_phys)
    return (
        path,
        start_point,
        goal_point,
        sdf_func,
        velocity_retina,
        X,
        Y,
        time,
        ratio,
        parameters,
    )


def obstacle_and_path(
    config_par_path,
    goal_point=None,
    path_method=None,
    file_to_config_path=None,
    type = '',
    path_to_config_path=None,
    ):
    start_point = config_par_path['start_point']
    B = config_par_path['B']
    c = config_par_path['c']
    heuristic_weight = config_par_path['heuristic_weight']
    pow_al = config_par_path['pow_al']
    pow_v0 = config_par_path['pow_v0']
    description = config_par_path['description']
    res_factor = config_par_path['res_factor']
    weight_sdf = config_par_path['weight_sdf']
    max_radius = config_par_path['max_radius']
    ratio = config_par_path['ratio']
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if path_to_config_path is not None:
        path, _, _, sdf_function, _, X_new, Y_new = load_config_path(
            path_to_config_path
        )
    else:
        # If config_path_fmm is None, all other parameters must be provided
        print("Creation of config file...")
        config_path_a = {}


        start_point = (start_point[0] * physical_width, start_point[1] * physical_height)
        goal_point = (goal_point[0] * physical_width, goal_point[1] * physical_height)

        # Reduce the cell size by a factor : res_factor
        grid_size = (len(x_phys),len(y_phys))
        
        parameters = {
            "start_point": tuple(float(x) for x in start_point),
            "goal_point": tuple(float(x) for x in goal_point),
            "grid_size": grid_size,
            "B": B,
            "heuristic_weight": heuristic_weight,
            "weight_sdf": weight_sdf,
            "c": c,
            "pow_v0": pow_v0,
            "pow_al": pow_al,
            "max_radius":max_radius,
            "method": path_method,
            "current_time": current_time,
            "type" : type,
            "description": description,
        }
        config_path_a["parameters"] = parameters

        save_path_path = f"data/retina2D_path_time_{current_time}.npy"

        if path_method == "astar":
            # Compute v0,vx and vy on this new domain with a certain size of cell
            v0, vx, vy, save_path_phi, save_path_flow = compute_v(
                x_phys, y_phys, velocity_retina, B, grid_size, ratio, sdf_func, c
            )
            
            if type =='free' :
                v0 = np.ones_like(v0)
                vx = None
                vy = None
                heuristic_weight = 0.0
                max_radius=5
            if type =='v1':
                v0 = np.ones_like(v0)
                pow_v0 = 0
                pow_al = 0
                heuristic_weight = 0.0
                max_radius=5

            path, travel_time = astar_anisotropic(
                x_phys,
                y_phys,
                v0,
                vx,
                vy,
                start_point,
                goal_point,
                sdf_func,
                heuristic_weight=heuristic_weight,
                pow_v0=pow_v0,
                pow_al=pow_al,
                max_radius=max_radius
            )

            path = np.array(path)  # de forme (N, 2)
            dist = np.array([abs(path[i + 1] - path[i]) for i in range(len(path) - 1)])
            n = ceil(np.max(dist) / (5 * 1e-3))
            if n > 1:
                path,distances = resample_path(path, len(path) * n)
        config_path_a["path_path"] = save_path_path
        config_path_a["path_phi"] = save_path_phi
        config_path_a["path_flow"] = save_path_flow
        config_path_a['distances'] = tuple(float(x) for x in distanes)
        config_path_a = {k: float(v) if isinstance(v, np.float32) else v for k, v in config_path_a.items()}

        np.save(save_path_path, path, allow_pickle=False)

        path_to_config_path = os.path.join(
            file_to_config_path, f"config_path__{path_method}_{current_time}.json"
        )
        with open(path_to_config_path, "w") as f:
            json.dump(config_path_a, f, indent=4)
            print("Config saved ")

    X, Y = np.meshgrid(x_phys, y_phys)
    obstacle_contour = contour_2D(sdf_func, X, Y, scale)

    return (
        np.array(start_point),
        np.array(goal_point),
        sdf_func,
        path,
        obstacle_contour,
        velocity_retina,
        current_time,
    )

def create_all_path(config_par_path,nb_points,ratio=5,failure_cases=False):


    file_to_config_path_pre_list = []
    file_to_config_path_pre = 'config_path/velocity_ratio_5/7'
    dir_config_path = Path(file_to_config_path_pre)

    failure_case_file = 'config_path/velocity_ratio_5/failure_case.json'
    
    
    
    if failure_cases :
        with open(failure_case_file,'r') as f:
            failure_case = json.load(f)
        print(failure_case)
        for item in dir_config_path.rglob("*.json"):
            file_name_path = os.path.join(dir_config_path, item.name)
            if file_name_path in failure_case:
                file_to_config_path_pre_list.append(file_name_path)
                
        goal_points=[]     
        for config_path in file_to_config_path_pre_list:
            with open(config_path, "r") as f:
                config_path_a_star = json.load(f)
            parameters_bis = config_path_a_star["parameters"]
            goal_points.append(list(np.array(parameters_bis['goal_point'])/20))
            
    for item in dir_config_path.rglob("*.json"):
            file_name_path = os.path.join(dir_config_path, item.name)
            file_to_config_path_pre_list.append(file_name_path)
    
    compute_goal_points = True
    if compute_goal_points:
        file_to_config_path_g = f"config_path/velocity_ratio_{ratio}"

        file_to_config_path = str(create_numbered_run_folder(file_to_config_path_g))
        
        os.makedirs(file_to_config_path, exist_ok=True)
        goal_points = create_list_of_goal_point(30000, config_par_path['start_point'],ratio)
        print(len(goal_points))
        goal_points = goal_points[:min(nb_points,len(goal_points))]
        for nb,goal_point in enumerate(goal_points):  
            print(f'Iter : {nb+1} over {len(goal_points)}')  
            goal_point = tuple(goal_point)
            (
                p_0,
                p_target,
                sdf_func,
                path,
                obstacle_contour,
                velocity_retina,
                current_time,
            ) = obstacle_and_path(
                config_par_path,
                path_method="astar",
                goal_point=goal_point,
                file_to_config_path=file_to_config_path,
                path_to_config_path=None,
            )
    return file_to_config_path

def create_list_of_goal_point(n, start_point,ratio):
    point_list = np.random.rand(n, 2)
    sdf_function, _ = sdf_func_and_velocity_func((1, 1), ratio)
    goal_points = [
        point
        for point in point_list
        if (
            sdf_function(point) < -0.4
            and point[0] < 0.9
            and np.linalg.norm(point - start_point) > 0.05
        )
    ]
    return goal_points



if __name__ == "__main__":
    obstacle_type = "retina"
    # agents_file = []
    # directory_path = Path("agents/")
    # for item in directory_path.iterdir():
    #     if item.is_dir() and "agent_TD3" in item.name:
    #         if "2025-04-23" in item.name or "2025-04-22" in item.name:
    #             agents_file.append(os.path.join(directory_path, item.name))
    
    agents_file = ["agents/agent_TD3_2025-05-07_15-48"]

    print("Number of agents : ", len(agents_file))

    with open("src/config_path_eva.yaml", "r") as f:
        config_par_path = yaml.safe_load(f)
          
    # file_to_config_path = create_all_path(config_par_path,500)
    ratio = 5
    sdf_func,velocity_retina,x_phys,y_phys,physical_width,physical_height,scale= load_sim_sdf(ratio)

    
    # file_to_config_path_g = f"config_path/velocity_ratio_{ratio}"
    # file_to_config_path = str(create_numbered_run_folder(file_to_config_path_g))
    file_to_config_path = 'config_path/velocity_ratio_5/42'
    # types = ['']
    
    # for type in types :
    #     obstacle_and_path(
    #         config_par_path,
    #         goal_point = (10.79606786617848/20,12.296130605776128/20),
    #         path_method="astar",
    #         file_to_config_path=file_to_config_path,
    #         type = type,
    #     )
        
    start_time_eva = time.time()
    list_config_paths = []
    dir_config_path = Path(file_to_config_path)

    for item in dir_config_path.rglob("*.json"):
        list_config_paths.append(os.path.join(dir_config_path, item.name))


    print("Number of path : ",len(list_config_paths))
    
    X,Y = np.meshgrid(x_phys,y_phys)
    obstacle_contour = contour_2D(sdf_func, X, Y, scale)
    print("Path generated - Go for evaluation")
    results = evaluate_after_training(
        agents_file,
        obstacle_contour=obstacle_contour,
        obstacle_type=obstacle_type,
        velocity_func=velocity_retina,
        sdf_func=sdf_func,
        list_config_paths=list_config_paths,
        sigma= config_par_path['sigma'],
        file_path_result = None,
    )
    end_time_eva = time.time()
    elapsed_time = (end_time_eva - start_time_eva) / 60
    print("Execution time:", elapsed_time, "minutes")
    # rank_agents_by_rewards(results)
