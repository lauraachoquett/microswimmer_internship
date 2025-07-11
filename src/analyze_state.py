import pickle
import numpy as np
import os
from pathlib import Path

import copy
from src.env_swimmer import MicroSwimmer
import pickle
from src.TD3 import TD3
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D


def initialize_parameters(agent_file, p_target, p_0, nb_points_path):
    path_config = os.path.join(agent_file, "config.pkl")
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    config_eval = copy.deepcopy(config)

    config_eval["random_helix"] = (
        config["random_helix"] if "random_helix" in config else False
    )
    config_eval["p_target"] = p_target
    config_eval["p_0"] = p_0
    config_eval["x_0"] = p_0
    config_eval["nb_points_path"] = nb_points_path
    config_eval["t_max"] = 12
    config_eval["eval_episodes"] = 4
    config_eval["velocity_bool"] = (
        config["velocity_bool"] if "velocity_bool" in config else False
    )
    config_eval["paraview"] = False
    config_eval["velocity_ahead"] = (
        config["velocity_ahead"] if "velocity_ahead" in config else False
    )
    config_eval["add_action"] = (
        config["add_action"] if "add_action" in config else False
    )
    config_eval["Dt_action"] = (
        config_eval["Dt_action"] if "Dt_action" in config else 1 / 30
    )
    config_eval["U"] = (
        config_eval["U"] if "U" in config else 1
    )
    config_eval['U']=1/10
    config_eval["gamma"] = (
        config_eval["gamma"] if "gamma" in config else 0
    )
    config_eval["n_lookahead"] = config["n_lookahead"] if "n_lookahead" in config else 5
    config_eval['dim']=3
    maximum_curvature = 30
    l = 1 / maximum_curvature
    Dt_action = 1 / maximum_curvature
    threshold = 0.5
    D = threshold**2 / (20 * Dt_action)
    config_eval["D"] = D
    return config_eval


filename = 'data/states.pkl'
with open(filename, 'rb') as f:
    states = pickle.load(f)
    states = np.array(states)
    


scale_pos = np.linspace(1,500,50,dtype=int)

def states_scaled(agents_file):
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['#1f77b4', 
          '#ff7f0e', 
          '#2ca02c', 
          '#d62728', 
          '#9467bd'] 
    legend_elements = []

    for id_agent,agent_name in enumerate(agents_file) : 
        print('Agent name :',agent_name)
        save_path_eval = os.path.join(agent_name,'eval_bg/')
        os.makedirs(save_path_eval,exist_ok=True)
                
        p_0 = np.array([0,0,0])
        p_target = p_0 + np.array([5,0,0])
        nb_points_path = 5000

        config_eval = initialize_parameters(agent_name, p_target, p_0, nb_points_path)
        config_eval_v = copy.deepcopy(config_eval)
        nb_points_path = config_eval_v["nb_points_path"]

        config_eval_v["D"] = 0
        config_eval_v["x_0"] = p_0

        env = MicroSwimmer(
            x_0=config_eval_v["x_0"],
            C=config_eval_v["C"],
            Dt=config_eval_v["Dt_action"],
            U=config_eval_v['U'],
            velocity_bool=config_eval_v["velocity_bool"],
            n_lookahead=config_eval_v['n_lookahead'],
            velocity_ahead=config_eval_v['velocity_ahead'],
            add_action=config_eval_v['add_action'],
            dim=config_eval_v['dim']
        )
        add_previous_action=False
        state_dim = env.observation_space.shape[0]
        if state_dim//3 ==8 and states.shape[1]==7:
            add_previous_action=True
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = TD3(state_dim, action_dim, max_action)
        policy_file = os.path.join(agent_name, "models/agent")
        agent.load(policy_file)

        past_action = np.array([1, 0.00000000e+00, 0.00000000e+00])
        past_action = past_action[None, :]  # shape (1, 3)

        action_list_per_scale = {}
        for scale in tqdm(scale_pos):
            action_list=[]
            for state in states :
                if add_previous_action:
                    state = np.concatenate((state,past_action),axis=0)
                pos = state[0]/scale
                state[0] = pos
                action = agent.select_action(state)
                action_list.append(action[0])
            action_list_per_scale[scale]=np.array(action_list)
            


        means = []
        mins = []
        maxs = []

        for scale in scale_pos:
            actions_list = action_list_per_scale[scale]
            actions_array = np.array(actions_list)
            
            means.append(np.mean(actions_array))
            mins.append(np.min(actions_array))
            maxs.append(np.max(actions_array))


        max_mean = np.max(np.array(means))
        gap = 1-max_mean
        print("Gap with ideal action : ",gap)
        scale_pos_array = np.array(scale_pos)

        # Moyenne
        means = np.ones_like(means)-means
        mins = np.ones_like(means)-mins
        maxs = np.ones_like(means)-maxs
        axs.plot(scale_pos_array, means, color=colors[id_agent], marker='o')

        # Min-Max en zone remplie
        axs.fill_between(scale_pos_array, mins, maxs, color=colors[id_agent], alpha=0.4)

        # Axes et style
        axs.set_xlabel('Scale')
        axs.set_ylabel('Action[0]')
        axs.set_yscale('log')
        axs.set_ylim(0.0001, 1)
        axs.set_title(f'Action[0] vs Scale (mean, min, max)')
        axs.legend()
        axs.grid(True)
        legend_elements.append(Line2D([0], [0], color=colors[id_agent], marker='o',
                                label=f'Agent {id_agent}'))


    # Sauvegarde
    axs.legend(handles=legend_elements)
    plt.tight_layout()
    # plt.savefig(os.path.join(save_path_eval, 'pos_scaled_action_summary_log.png'), dpi=300, bbox_inches='tight')
    plt.savefig('fig/pos_scaled_action_summary_log.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    agents_file = ['agents/agent_TD3_2025-07-11_15-11','agents/agent_TD3_2025-07-10_16-44','agents/agent_TD3_2025-05-21_16-55','agents/agent_TD3_2025-07-10_10-33']
    states_scaled(agents_file)
            
            
        