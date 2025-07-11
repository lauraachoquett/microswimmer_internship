import os
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import copy
from src.generate_path import *
from scipy.spatial import KDTree
from src.env_swimmer import MicroSwimmer
from src.evaluate_agent import evaluate_agent
from src.plot import plot_trajectories
import pickle
from src.plot import hist_scientific
from src.TD3 import TD3
import plotly.graph_objects as go


def visualize_streamline(
    agent_name,
    file_name_or,
    save_path_eval,
    type="",
    title="",
    parameters=[],
    offset=0.2,
):
    save_path_streamline = os.path.join(save_path_eval, "streamlines/")
    if not os.path.exists(save_path_streamline):
        os.makedirs(save_path_streamline)
        
    p_0 = np.array([0,0,0])
    
    p_target = p_0 + np.array([5,0,0])
    nb_points_path = 5000

    config_eval = initialize_parameters(agent_name, p_target, p_0, nb_points_path)
    config_eval_v = copy.deepcopy(config_eval)

    trajectories = {}
    nb_points_path = config_eval_v["nb_points_path"]

    nb_starting_point = 8
    p_0_above = p_0 + np.array([0 ,offset, 0])
    p_target_above = p_target + np.array([0, offset,0])
    p_0_below = p_0 + np.array([0, -offset,0])
    p_target_below = p_target + np.array([0, -offset,0])
    path, _ = generate_simple_line(p_0, p_target, nb_points_path)
    print("Difference between points :",path[1]-path[0])
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(path[:, 0], path[:, 1], label="path", color="black", linewidth=2, zorder=10)


    # Force la figure à remplir la taille figsize même si les axes sont déformés
    
    trajectories["path"] = path
    theta_list = []

    for starting_point in path_starting_point:
        config_eval_v["x_0"] = starting_point
        env = MicroSwimmer(x_0 =config_eval_v["x_0"] , C = config_eval_v["C"] , Dt = config_eval_v["Dt_action"] , U=config_eval_v['U'], velocity_bool  = config_eval_v["velocity_bool"], n_lookahead = config_eval_v['n_lookahead'],velocity_ahead = config_eval_v['velocity_ahead'],add_action = config_eval_v['add_action'],dim = config_eval_v['dim'])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = TD3(state_dim, action_dim, max_action)

        policy_file = os.path.join(agent_name, "models/agent")
        agent.load(policy_file)
        _, _, _, _, states_list_per_episode,actions = evaluate_agent(
            agent=agent,
            env=env,
            eval_episodes=1,
            config=config_eval_v,
            save_path_result_fig=save_path_streamline,
            file_name=file_name_or,
            random_parameters=False,
            title="",
            plot=True,
            parameters=parameters,
            plot_background=False,
        )
        actions = np.array(actions).squeeze(0)
        for i in range(actions.shape[0]-1):
            a1 = actions[i]
            a2 = actions[i + 1]
            norm1 = np.linalg.norm(a1)
            norm2 = np.linalg.norm(a2)
            if norm1 == 0 or norm2 == 0:
                continue
            cos_theta = np.dot(a1, a2) / (norm1 * norm2)
            theta_list.append(cos_theta)
        trajectories[f"{starting_point}"] = states_list_per_episode
        ax.scatter(config_eval_v["x_0"][0], config_eval_v["x_0"][1], color="red", label="start points")
        plot_trajectories(ax, states_list_per_episode, path, title="streamlines")
        
    ax.set_aspect('auto', adjustable='box')
    ax.plot(0.05*np.ones_like(len(path)))
    ax.plot(-0.05*np.ones_like(len(path)))

    pkl_save_directory = os.path.join(save_path_streamline, type)
    os.makedirs(pkl_save_directory, exist_ok=True)
    pkl_save_path = os.path.join(pkl_save_directory, f"{file_name_or}_trajectories.pkl")
    with open(pkl_save_path, "wb") as f:
        pickle.dump(trajectories, f)
    directory_fig = os.path.join(save_path_streamline, type)
    os.makedirs(directory_fig, exist_ok=True)
    path_save_fig = os.path.join(directory_fig, file_name_or + ".png")
    fig.savefig(path_save_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    hist_scientific(theta_list,xlabel=r'$\cos(\theta)$',title='',save_name=os.path.join(directory_fig,'cos_theta_hist'),bins=20)
    
def test_state_action_pos(
    agent_name,
    save_path_eval,
):
    save_path_streamline = os.path.join(save_path_eval, "streamlines/")
    if not os.path.exists(save_path_streamline):
        os.makedirs(save_path_streamline)
        
    p_0 = np.array([0,0,0])
    p_target = p_0 + np.array([5,0,0])
    nb_points_path = 5000

    config_eval = initialize_parameters(agent_name, p_target, p_0, nb_points_path)
    config_eval_v = copy.deepcopy(config_eval)
    nb_points_path = config_eval_v["nb_points_path"]

    path, _ = generate_simple_line(p_0, p_target, nb_points_path)
    config_eval_v["path"] = path
    config_eval_v["tree"] = KDTree(path)
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)

    policy_file = os.path.join(agent_name, "models/agent")
    agent.load(policy_file)

    # Scan pos[0], pos[1], pos[2] over several orders of magnitude between 1e-6 and 1e-1
    pos_range = np.logspace(-6, -1, 50)
    results = []
    for x in pos_range:
        for y in pos_range:
            for z in pos_range:
                pos = np.array([x, y, z])
                state = np.array([
                    pos,
                    [1.00000000e-01, -8.46208763e-08, 8.03898274e-07],
                    [5.00005000e-05, 0.00000000e+00, 0.00000000e+00],
                    [1.00001000e-04, 0.00000000e+00, 0.00000000e+00],
                    [1.50001500e-04, 0.00000000e+00, 0.00000000e+00],
                    [2.00002000e-04, 0.00000000e+00, 0.00000000e+00],
                    [2.50002500e-04, 0.00000000e+00, 0.00000000e+00]
                ])
                action = agent.select_action(state)
                if action[0] > 0.995:
                    results.append((x, y, z, action[0]))
    results = np.array(results)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if results.shape[0] > 0:
        sc = ax.scatter(results[:, 0], results[:, 1], results[:, 2], c=results[:, 3], cmap='Reds', s=40,alpha=0.5)
        plt.colorbar(sc, ax=ax, label='action[0]')
        ax.set_title("Positions où action[0] > 0.99")
        # Statistiques sur x, y, z
        for i, name in enumerate(['x', 'y', 'z']):
            vals = results[:, i]
            min_val = abs(vals.min())
            max_val =  abs(vals.max())
            if min_val > 0 and max_val > 0:
                sig_digits = int(np.floor(np.log10(max_val)) - np.floor(np.log10(min_val)))
            else:
                sig_digits = 'N/A'
            print(f"{name}: min={min_val:.2e}, max={max_val:.2e}, significant digits={sig_digits}")
    else:
        ax.text2D(0.5, 0.5, "Aucune position trouvée où action[0] > 0.99", transform=ax.transAxes, ha='center')
    ax.set_xlabel('log10(v_x)')
    ax.set_ylabel('log10(v_y)')
    ax.set_zlabel('log10(v_z)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path_eval,'test_action_pos.png'),dpi=200,bbox_inches='tight')

    # Interactive 3D plot with plotly
    try:
        import plotly.graph_objects as go
        if results.shape[0] > 0:
            fig_plotly = go.Figure(data=[go.Scatter3d(
                x=results[:, 0],
                y=results[:, 1],
                z=results[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=results[:, 3],
                    colorscale='Reds',
                    colorbar=dict(title='action[0]'),
                    opacity=0.8
                ),
                text=[f"action[0]={a:.3f}" for a in results[:, 3]]
            )])
            fig_plotly.update_layout(
                scene=dict(
                    xaxis_title='v_x',
                    yaxis_title='v_y',
                    zaxis_title='v_z'
                ),
                title="Positions où action[0] > 0.99 (interactive)",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            interactive_path = os.path.join(save_path_eval, 'test_action_pos_interactive.html')
            fig_plotly.write_html(interactive_path)
            print(f"Interactive 3D plot saved to {interactive_path}")
    except ImportError:
        print("Plotly is not installed. Install it with 'pip install plotly' for interactive 3D visualization.")
        
    # Affichage d'autres infos pertinentes
    if results.shape[0] > 0:
        print(f"Nombre total de points où action[0] > 0.99 : {results.shape[0]}")
        print(f"action[0] min={results[:,3].min():.3f}, max={results[:,3].max():.3f}, mean={results[:,3].mean():.3f}, std={results[:,3].std():.3f}")
        
def test_state_action_vel(
    agent_name,
    save_path_eval,
):
    save_path_streamline = os.path.join(save_path_eval, "streamlines/")
    if not os.path.exists(save_path_streamline):
        os.makedirs(save_path_streamline)
        
    p_0 = np.array([0,0,0])
    p_target = p_0 + np.array([5,0,0])
    nb_points_path = 5000

    config_eval = initialize_parameters(agent_name, p_target, p_0, nb_points_path)
    config_eval_v = copy.deepcopy(config_eval)
    nb_points_path = config_eval_v["nb_points_path"]

    path, _ = generate_simple_line(p_0, p_target, nb_points_path)
    config_eval_v["path"] = path
    config_eval_v["tree"] = KDTree(path)
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)

    policy_file = os.path.join(agent_name, "models/agent")
    agent.load(policy_file)

    # Scan pos[0], pos[1], pos[2] over several orders of magnitude between 1e-6 and 1e-1
    pos_range = np.logspace(-6, 0, 50)
    results = []
    for v_x in pos_range:
        for v_y in pos_range:
            for v_z in pos_range:
                vel = np.array([v_x, v_y, v_z])
                pos = np.array([5.31e-03,2.80e-03,2.58e-03])
                state = np.array([
                    pos,
                    vel,
                    [5.00005000e-05, 0.00000000e+00, 0.00000000e+00],
                    [1.00001000e-04, 0.00000000e+00, 0.00000000e+00],
                    [1.50001500e-04, 0.00000000e+00, 0.00000000e+00],
                    [2.00002000e-04, 0.00000000e+00, 0.00000000e+00],
                    [2.50002500e-04, 0.00000000e+00, 0.00000000e+00]
                ])
                action = agent.select_action(state)
                if action[0] > 0.99:
                    results.append((v_x, v_y, v_z, action[0]))
    results = np.array(results)
    # Matplotlib 3D static plot (as before)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if results.shape[0] > 0:
        sc = ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                        c=results[:, 3], cmap='Reds', s=50, alpha=0.5)
    
        plt.colorbar(sc, ax=ax, label='action[0]')
        ax.set_title("Zones (log) où action[0] > 0.99")
        for i, name in enumerate(['v_x', 'v_y', 'v_z']):
            vals = results[:, i]
            min_val = abs(vals.min())
            max_val = abs(vals.max())
            if min_val > 0 and max_val > 0:
                sig_digits = int(np.floor(np.log10(max_val)) - np.floor(np.log10(min_val)))
            else:
                sig_digits = 'N/A'
            print(f"{name}: min={min_val:.2e}, max={max_val:.2e}, significant digits={sig_digits}")
    else:
        ax.text2D(0.5, 0.5, "Aucune position trouvée où action[0] > 0.99", transform=ax.transAxes, ha='center')
    ax.set_xlabel('log10(v_x)')
    ax.set_ylabel('log10(v_y)')
    ax.set_zlabel('log10(v_z)')    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_eval, 'test_action_vel.png'), dpi=200, bbox_inches='tight')

    # Interactive 3D plot with plotly
    try:
        import plotly.graph_objects as go
        if results.shape[0] > 0:
            fig_plotly = go.Figure(data=[go.Scatter3d(
                x=results[:, 0],
                y=results[:, 1],
                z=results[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=results[:, 3],
                    colorscale='Reds',
                    colorbar=dict(title='action[0]'),
                    opacity=0.8
                ),
                text=[f"action[0]={a:.3f}" for a in results[:, 3]]
            )])
            fig_plotly.update_layout(
                scene=dict(
                    xaxis_title='v_x',
                    yaxis_title='v_y',
                    zaxis_title='v_z'
                ),
                title="Positions où action[0] > 0.99 (interactive)",
                margin=dict(l=0, r=0, b=0, t=40)
            )
            interactive_path = os.path.join(save_path_eval, 'test_action_vel_interactive.html')
            fig_plotly.write_html(interactive_path)
            print(f"Interactive 3D plot saved to {interactive_path}")
    except ImportError:
        print("Plotly is not installed. Install it with 'pip install plotly' for interactive 3D visualization.")

    # Affichage d'autres infos pertinentes
    if results.shape[0] > 0:
        print(f"Nombre total de points où action[0] > 0.99 : {results.shape[0]}")
        print(f"action[0] min={results[:,3].min():.3f}, max={results[:,3].max():.3f}, mean={results[:,3].mean():.3f}, std={results[:,3].std():.3f}")
        
    pos = np.array([5.31e-03,2.80e-03,2.58e-03])
        
    vel = np.array([0.00708938,-0.01071362,-0.00856874])*100
    state = np.array([
        pos,
        vel,
        [5.00005000e-05, 0.00000000e+00, 0.00000000e+00],
        [1.00001000e-04, 0.00000000e+00, 0.00000000e+00],
        [1.50001500e-04, 0.00000000e+00, 0.00000000e+00],
        [2.00002000e-04, 0.00000000e+00, 0.00000000e+00],
        [2.50002500e-04, 0.00000000e+00, 0.00000000e+00]
    ])
    action = agent.select_action(state)
    print('Action :',action)
    
    vel = np.array([0.00708938,-0.01071362,-0.00856874])*10
    state = np.array([
        pos,
        vel,
        [5.00005000e-05, 0.00000000e+00, 0.00000000e+00],
        [1.00001000e-04, 0.00000000e+00, 0.00000000e+00],
        [1.50001500e-04, 0.00000000e+00, 0.00000000e+00],
        [2.00002000e-04, 0.00000000e+00, 0.00000000e+00],
        [2.50002500e-04, 0.00000000e+00, 0.00000000e+00]
    ])
    action = agent.select_action(state)
    print("Action : ", action)
    
    vel = np.array([0.00708938,-0.01071362,-0.00856874])*1
    state = np.array([
        pos,
        vel,
        [5.00005000e-05, 0.00000000e+00, 0.00000000e+00],
        [1.00001000e-04, 0.00000000e+00, 0.00000000e+00],
        [1.50001500e-04, 0.00000000e+00, 0.00000000e+00],
        [2.00002000e-04, 0.00000000e+00, 0.00000000e+00],
        [2.50002500e-04, 0.00000000e+00, 0.00000000e+00]
    ])
    action = agent.select_action(state)
    print("Action : ", action)

    

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


if __name__ == "__main__":
    agents_file = []

    agent_name = 'agents/agent_TD3_2025-07-10_10-33'
    save_path_eval = os.path.join(agent_name,'eval_bg/')
    file_name_or ='streamlines'
    type='line'
    visualize_streamline(
        agent_name,
        file_name_or,
        save_path_eval,
        type=type,
        title="",
        parameters=[],
        offset=0.05,
    )
    print("TEST ACTION VELOCITY")
    test_state_action_vel(
        agent_name,
        save_path_eval,
    )
    
    print("TEST ACTION POSITION")
    test_state_action_pos(
        agent_name,
        save_path_eval,
    )