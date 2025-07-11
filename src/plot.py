import json
import os
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import plotly.graph_objects as go
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour l'import
import pandas as pd

colors_default = plt.cm.tab10.colors
from src.generate_path import (generate_curve, generate_demi_circle_path,
                               generate_random_ondulating_path)
from src.simulation import rankine_vortex, uniform_velocity


def video_trajectory(
    fig,
    ax,
    trajectory,
    path_save_video="",
    color_id=0,
    colors=plt.cm.tab10.colors,
):
    trajectory = np.array(trajectory)
    step = max(1, len(trajectory) // 1000)
    trajectory = trajectory[::step]
    print(trajectory.shape)
    # Initialisation de la ligne de trajectoire animée
    (line,) = ax.plot([], [], color=colors[color_id], linewidth=0.9)
    start_dot = ax.scatter([], [], color=colors[color_id], s=5)
    end_dot = ax.scatter([], [], color=colors[color_id], s=5)
    ax.set_aspect("equal")
    ax.set_axis_off()

    def init():
        line.set_data([], [])
        start_dot.set_offsets(np.empty((0, 2)))
        end_dot.set_offsets(np.empty((0, 2)))
        return line, start_dot, end_dot

    def update(i):
        x, y = trajectory[: i + 1, 0], trajectory[: i + 1, 1]
        line.set_data(x, y)
        start_dot.set_offsets([trajectory[0]])
        end_dot.set_offsets([trajectory[i]])
        return line, start_dot, end_dot

    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectory), init_func=init, blit=True, interval=20
    )

    ani.save(path_save_video, writer="ffmpeg", dpi=200)

    plt.close(fig)


def plot_trajectories(
    ax,
    trajectories_list,
    path,
    title,
    a=0,
    center=np.zeros(2),
    cir=0,
    dir=np.zeros(2),
    norm=0,
    plot_background=False,
    type="",
    color_id=0,
    colors=plt.cm.tab10.colors,
    label="",
):
    if colors is None:
        colors = colors_default
    if isinstance(trajectories_list[0][0], np.ndarray):
        for idx, list_state in enumerate(trajectories_list):

            states = list_state[0]

            color_id_t = max(idx, color_id)
            ax.plot(
                states[:, 0],
                states[:, 1],
                color=colors[color_id_t],
                linewidth=0.9,
                label=label,
            )
            ax.scatter(states[-1, 0], states[-1, 1], color=colors[color_id_t], s=5)
            ax.scatter(states[0, 0], states[0, 1], color=colors[color_id_t], s=5)
            ax.set_aspect("equal")
    else:
        states = np.array(trajectories_list[0])
        color_id_t = color_id
        ax.plot(
            states[:, 0],
            states[:, 1],
            color=colors[color_id_t],
            linewidth=0.9,
            label=label,
        )
        ax.scatter(states[-1, 0], states[-1, 1], color=colors[color_id_t], s=5)
        ax.scatter(states[0, 0], states[0, 1], color=colors[color_id_t], s=5)
        ax.set_aspect("equal")
    if plot_background:
        x_bound = ax.get_xlim()
        y_bound = ax.get_ylim()
        plot_background_velocity(type, x_bound, y_bound, a, center, cir, dir, norm)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title != "":
        ax.set_title(f"{title}")
    else:
        if type == "uniform":
            ax.set_title(f"Trajectories - norm : {norm}")
        if type == "rankine":
            ax.set_title(f"Trajectories - a : {a} - circulation : {cir}")


def plot_html_3d(
    trajectories_list, save_path_html, list_of_path,dir=None,center=None,a=None, obstacle_contour = None,colors=None, color_id=0
):
    if colors is None:
        colors = ["red", "blue", "green", "orange", "purple", "brown"]
    fig = go.Figure()
    for path in list_of_path:
        x = path[:, 0]
        y = path[:, 1]
        z = path[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name="Path",
                line=dict(color="black", width=4),
            )
        )

    if isinstance(trajectories_list[0][0], np.ndarray):
        for idx, list_state in enumerate(trajectories_list):
            states = list_state[0]
            color_id_t = max(idx, color_id)
            fig.add_trace(
                go.Scatter3d(
                    x=states[:, 0],
                    y=states[:, 1],
                    z=states[:, 2],
                    mode="lines",
                    name=f"Trajectory {idx}",
                    line=dict(color=colors[color_id_t], width=1),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[states[0, 0]],
                    y=[states[0, 1]],
                    z=[states[0, 2]],
                    mode="markers",
                    marker=dict(size=3, color=colors[color_id_t]),
                    name=f"Start {idx}",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[states[-1, 0]],
                    y=[states[-1, 1]],
                    z=[states[-1, 2]],
                    mode="markers",
                    marker=dict(size=3, color=colors[color_id_t]),
                    name=f"End {idx}",
                )
            )
    else:
        states = np.array(trajectories_list[0])
        color_id_t = color_id
        fig.add_trace(
            go.Scatter3d(
                x=states[:, 0],
                y=states[:, 1],
                z=states[:, 2],
                mode="lines",
                name="Trajectory",
                line=dict(color=colors[color_id_t], width=1),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[states[0, 0]],
                y=[states[0, 1]],
                z=[states[0, 2]],
                mode="markers",
                marker=dict(size=3, color=colors[color_id_t]),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[states[-1, 0]],
                y=[states[-1, 1]],
                z=[states[-1, 2]],
                mode="markers",
                marker=dict(size=3, color=colors[color_id_t]),
                name="End",
            )
        )
    
    if dir is not None and np.any(dir)!=0:
        start = np.array([-1/2, 0, 0])
        direction = dir 
        fig.add_trace(go.Cone(
            x=[start[0]],
            y=[start[1]],
            z=[start[2]],
            u=[direction[0]],
            v=[direction[1]],
            w=[direction[2]],
            sizemode="absolute",
            sizeref=0.15,
            anchor="tail",
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            name="Direction"
        ))
    if center is not None and np.any(center)!=0:
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers",
                marker=dict(size=3, color='black'),
                name=f"Center",
            )
        )

        x,y,z =draw_circle(center,a)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='black', width=4),
            name='vortex radius'
        ))
    
    if obstacle_contour is not None : 

        z = 0.5 * np.ones(obstacle_contour.shape[0])
        x = obstacle_contour[:,0]
        y = obstacle_contour[:,1]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                name="Contour",
                marker=dict(size=1, color='indianred'),
            )
        )

        
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
            aspectmode="data",
        ),
        title="Trajectoires 3D interactives",
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(save_path_html)


def plot_trajectories_3D(
    ax,
    trajectories_list,
    colors=None,
    label=None,
    title="",
    type="",
    norm=None,
    a=None,
    cir=None,
    color_id=0,
):
    colors_default = ["red", "blue", "green", "orange", "purple", "brown"]
    if colors is None:
        colors = colors_default

    if isinstance(trajectories_list[0][0], np.ndarray):
        for idx, list_state in enumerate(trajectories_list):
            states = list_state[0]
            color_id_t = max(idx, color_id)
            ax.plot(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                color=colors[color_id_t],
                linewidth=0.9,
                label=label,
            )
            ax.scatter(
                states[0, 0], states[0, 1], states[0, 2], color=colors[color_id_t], s=5
            )
            ax.scatter(
                states[-1, 0],
                states[-1, 1],
                states[-1, 2],
                color=colors[color_id_t],
                s=5,
            )
    else:
        states = trajectories_list
        color_id_t = color_id
        ax.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            color=colors[color_id_t],
            linewidth=0.9,
            label=label,
        )
        ax.scatter(
            states[0, 0], states[0, 1], states[0, 2], color=colors[color_id_t], s=5
        )
        ax.scatter(
            states[-1, 0], states[-1, 1], states[-1, 2], color=colors[color_id_t], s=5
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if title != "":
        ax.set_title(f"{title}")
    else:
        if type == "uniform":
            ax.set_title(f"Trajectories - norm : {norm}")
        if type == "rankine":
            ax.set_title(f"Trajectories - a : {a} - circulation : {cir}")


def plot_action(path, x, p_0, id_cp, action, id):
    plt.scatter(x[0], x[1], color=colors_default[id % 10])
    plt.annotate(
        f"{id}", xy=(x[0], x[1]), xytext=(x[0], x[1] + 1 / (64 * 20))  # texte en LaTeX
    )
    plt.scatter(
        path[id_cp, 0], path[id_cp, 1], color=colors_default[id % 10], marker="*"
    )
    plt.quiver(x[0], x[1], action[0], action[1], scale=20, width=0.005, color="grey")
    plt.xlabel("x")
    plt.ylabel("y")


def plot_background_velocity(
    type,
    x_bound,
    y_bound,
    a=0.25,
    center=(0.5, 0.5),
    cir=0.8,
    dir=np.zeros(2),
    norm=0.0,
):
    x = np.linspace(x_bound[0], x_bound[1], 10)
    y = np.linspace(y_bound[0], y_bound[1], 10)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if type == "rankine":
                v = rankine_vortex((X[i, j], Y[i, j]), a, center, cir)
                U[i, j] = v[0]
                V[i, j] = v[1]
            if type == "uniform":
                v = uniform_velocity(dir, norm)
                U[i, j] = v[0]
                V[i, j] = v[1]
    plt.quiver(X, Y, U, V, scale=15, width=0.002, color="gray")
    if type == "rankine":
        plt.scatter(center[0], center[1], marker="*")
    plt.xlabel("x")
    plt.ylabel("y")


def plot_success_rate(path_json_file, agent_file, save_plot):
    with open(path_json_file, "r") as f:
        results = json.load(f)
    result_agent = results[f"{agent_file}"]["results_per_config"]

    short_labels = []
    for k in result_agent.keys():
        name = k.split("__")[-1].replace(".json", "")  # Extrait '2025-05-06_17-26-52'
        short_labels.append(name)
    name = {}
    sucess_rate_d = {}
    for id, config in enumerate(result_agent.keys()):
        sucess_rate_d[id] = result_agent[config]["success_rate"]
        name[id] = short_labels[id]

    ids = list(sucess_rate_d.keys())
    values = list(sucess_rate_d.values())
    plt.figure(figsize=(10, 5))
    plt.bar(ids, values, tick_label=[f"{i}" for i in ids])
    plt.xlabel("path ID")
    plt.ylabel("Success rate")
    plt.title("Success rate")
    plt.grid(True)
    file_table = os.path.join(save_plot, "table.json")
    with open(file_table, "w") as f:
        json.dump(name, f, indent=4)
    file_plot = os.path.join(save_plot, f"{agent_file}_result_success_rate.png")
    os.makedirs(os.path.join(save_plot, "agents"), exist_ok=True)
    plt.savefig(file_plot, dpi=200, bbox_inches="tight")


def analyze_and_visualize_agent_data(
    data, output_dir="./results_evaluation", fig_dir="./fig", name_fig=""
):
    df = pd.json_normalize(data)
    training_columns = [
        col
        for col in df.columns
        if col.startswith("training type.") and not col.endswith("load_model")
    ]
    df["training_type_str"] = df[training_columns].apply(
        lambda row: ", ".join(
            [f"{col.split('.')[-1]}={row[col]}" for col in training_columns]
        ),
        axis=1,
    )

    random_curve_agents = df[
        df["training_type_str"].str.contains("random_curve=True", na=False)
    ]
    random_curve_file = os.path.join(output_dir, "agent_random_curve.json")
    os.makedirs(os.path.dirname(random_curve_file), exist_ok=True)
    random_curve_agents.to_json(random_curve_file, orient="records", indent=4)

    agent_counts = df["training_type_str"].value_counts().reset_index()
    agent_counts.columns = ["training_type", "agent_count"]
    print("Number of agents per training type:")
    print(agent_counts)

    agent_counts_file = os.path.join(output_dir, "agent_counts.json")
    agent_counts.to_json(agent_counts_file, orient="records", indent=4)
    print(f"Agent counts saved to {agent_counts_file}")

    filtered_training_types = agent_counts[agent_counts["agent_count"] >= 1][
        "training_type"
    ]
    df = df[df["training_type_str"].isin(filtered_training_types)]

    def format_training_type_label(training_type):
        label = training_type.split(", ")
        if "False" in label[0] or "False" in label[1]:
            label_bis = ["Free"]
        else:
            label_bis = ["Varying background"]
        if "True" in label[2]:
            label_bis.append("Varying Curve")
        else:
            label_bis.append("Circle")
        n = int(label[-1].split("=")[1])
        label_bis.append(f"Lookahead (n={n})")
        return "\n".join(label_bis)

    df_melted = df.melt(
        id_vars=["training_type_str"],
        value_vars=["mean_reward", "mean_reward_t", "mean_reward_d"],
        var_name="metric",
        value_name="reward",
    )

    df_melted = df_melted[df_melted["training_type_str"].isin(filtered_training_types)]

    df_melted["metric"] = df_melted["metric"].map(
        {
            "mean_reward": "Overall mean return",
            "mean_reward_t": r"Time return: $-C \sum \Delta t_{sim}$",
            "mean_reward_d": r"Distance return: $-\beta \sum d$",
        }
    )

    df_melted["training_label"] = df_melted["training_type_str"].apply(
        format_training_type_label
    )

    plt.figure(figsize=(16, 9))
    sns.set(style="whitegrid")

    ax = sns.stripplot(
        data=df_melted,
        x="training_label",
        y="reward",
        hue="metric",
        dodge=True,
        jitter=True,
        size=6,
        palette="Set2",
    )

    plt.rcParams["text.usetex"] = True
    plt.title("Rewards per training", fontsize=14)
    plt.xlabel("Training type", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    xticks = ax.get_xticks()
    for i in range(len(xticks) - 1):
        xpos = (xticks[i] + xticks[i + 1]) / 2
        ax.axvline(x=xpos, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()

    by_label = OrderedDict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=15,
        title_fontsize=17,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{name_fig}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")


def plot_return_beta(file_path):

    with open(file_path, "r") as file:
        data = json.load(file)

    # Extraire les valeurs de beta et des récompenses
    betas = []
    mean_rewards = []
    mean_rewards_t = []
    mean_rewards_d = []

    for entry in data:
        betas.append(entry["training type"]["beta"])
        mean_rewards.append(entry["mean_reward"])
        mean_rewards_t.append(entry["mean_reward_t"])
        mean_rewards_d.append(entry["mean_reward_d"])

    # Tracer les récompenses en fonction de beta
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2")
    plt.scatter(
        betas, mean_rewards, label="Overall mean return", marker="o", color=palette[0]
    )
    plt.scatter(
        betas,
        mean_rewards_t,
        label=r"Time return: $-C \sum \Delta t_{sim}$",
        marker="o",
        color=palette[1],
    )
    plt.scatter(
        betas,
        mean_rewards_d,
        label=r"Distance return: $-\beta \sum d$",
        marker="o",
        color=palette[2],
    )

    # Ajouter des labels et une légende
    plt.xlabel("Beta")
    plt.ylabel("Reward")
    plt.legend(
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    plt.grid(True)

    # Afficher le graphique
    plt.savefig("fig/rank_beta_return.png", dpi=200, bbox_inches="tight")


def paraview_export(path_physical,  output_save_path,trajectories=None):
    os.makedirs(output_save_path,exist_ok=True)
    if path_physical is not None and len(path_physical) > 0:
        try:
            print(f"Path points shape: {path_physical.shape}")
            if path_physical.shape[1] == 3:
                path_polydata = pv.PolyData(path_physical)
                
                if len(path_physical) > 1:
                    lines = []
                    for i in range(len(path_physical) - 1):
                        lines.extend([2, i, i + 1])  
                    path_polydata.lines = np.array(lines)
                    print(f"Nombre de segments créés: {len(path_physical) - 1}")
                
                path_output = os.path.join(output_save_path,'path.vtp')
                path_polydata.save(path_output)
                print(f"Chemin sauvegardé: {path_output}")
                
        except Exception as e:
            print(f"Erreur lors de la création du chemin: {e}")
    if trajectories is not None:
        for id,elt in enumerate(trajectories) :
            output_save_path_agent = os.path.join(output_save_path,f'agent_{id}')
            os.makedirs(output_save_path_agent,exist_ok=True)
            all_vtp_filenames=[]
            trajectory = elt[0]
            if trajectory is not None and len(trajectory) > 0:
                try:
                    if trajectory.shape[1] == 3:
                        for t, point in enumerate(trajectory):
                            point_cloud = pv.PolyData(np.array([point]))  # Single point
                            filename = f"agent_{t:04d}_trajectory_{id}.vtp"
                            full_path = os.path.join(output_save_path_agent, filename)
                            point_cloud.save(full_path)
                            all_vtp_filenames.append((t, filename))
                        print(f"{len(trajectory)} fichiers .vtp générés pour l'agent {id}.")
                        
                        
                except Exception as e:
                    print(f"Erreur lors de la création du chemin: {e}")
                    print(f"Format attendu: liste de [i,j,k] où i,j,k sont des indices de grille")
                        
            pvd_path = os.path.join(output_save_path_agent, f"agent_{id}.pvd")

            with open(pvd_path, "w") as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
                f.write('  <Collection>\n')
                for t, filename in all_vtp_filenames:
                    f.write(f'    <DataSet timestep="{t}" group="" part="0" file="{filename}"/>\n')
                f.write('  </Collection>\n')
                f.write('</VTKFile>\n')

            print(f"Fichier PVD créé : {pvd_path}")
            
        
def draw_circle(center,radius):
    t = np.linspace(0,2*np.pi,200)
    print(center)
    x = center[0] * np.ones_like(t) + radius * np.cos(t)
    y = center[1] * np.ones_like(t) + radius * np.sin(t)
    z = np.ones_like(x) * center[-1]
    circle = np.stack((x,y,z),axis=1)
    return x,y,z

def plot_success_rate_D_state(file, title):
    length_scale = 0.269 / 20
    ms_length = 15
    import matplotlib.ticker as ticker

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'hspace': 0.25})

    markers = ['o', 'x']  # rond pour le premier fichier, croix pour le second
    labels = ['Training 1','Training 2']
    for idx, path in enumerate(file):
        with open(path, 'r') as f:
            data = json.load(f)

        agent = list(data.keys())[0]
        results = data[agent]

        D_states = []
        success_rates = []
        mean_rewards_time = []
        mean_rewards_distance = []

        for D_state, infos in results.items():
            D_states.append(float(D_state))
            config = list(infos["results_per_config"].values())[0]
            success_rates.append(config["success_rate"])
            rewards_time = config.get("rewards_time", [])
            rewards_distance = config.get("rewards_distance", [])
            mean_rewards_time.append(np.mean(rewards_time) if rewards_time else np.nan)
            mean_rewards_distance.append(np.mean(rewards_distance) if rewards_distance else np.nan)

        D_states = (np.array(D_states) / length_scale) / ms_length
        # Sort by D_state for nice plots
        D_states, success_rates, mean_rewards_time, mean_rewards_distance = zip(
            *sorted(zip(D_states, success_rates, mean_rewards_time, mean_rewards_distance))
        )

        marker = markers[idx % len(markers)]
        label_fig = labels[idx%len(labels)]

        # Plot 1: Success rate
        axs[0].plot(D_states, success_rates, linestyle='-', marker=marker,label=label_fig)
        axs[0].set_ylabel("Success rate", fontsize=12)
        axs[0].grid(True, which='both', linestyle='--', alpha=0.6)
        axs[0].set_ylim(0, 1.05)

        # Plot 2: Mean rewards_time
        axs[1].plot(D_states, mean_rewards_time, linestyle='-', marker=marker,label=label_fig)
        axs[1].set_ylabel(r"$\bar{J}_t$", fontsize=12)
        axs[1].grid(True, which='both', linestyle='--', alpha=0.6)

        # Plot 3: Mean rewards_distance
        axs[2].plot(D_states, mean_rewards_distance, linestyle='-', marker=marker,label=label_fig)
        axs[2].set_xlabel(r" Noise magnitude (% of swimmer length)", fontsize=12)
        axs[2].set_ylabel(r"$\bar{J}_d$", fontsize=12)
        axs[2].grid(True, which='both', linestyle='--', alpha=0.6)

    # Scientific style
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join('fig', title)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def hist_scientific(data, xlabel='', ylabel='Relative Frequency', title='', 
                   bins='auto', save_name=None):

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif'],
    })
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Use density=True to get relative frequencies (area sums to 1)
    counts, bin_edges, patches = ax.hist(
        data, bins=bins, 
        alpha=0.7, 
        color='steelblue', 
        edgecolor='black', 
        linewidth=0.8,
        density=True
    )
    
    # Optionally, scale to sum to 1 (relative frequency per bin, not density)
    n = len(data)
    print("Largeur des bins :",np.diff(bin_edges))
    rel_freq = counts * np.diff(bin_edges)
    ax.clear()
    ax.bar(bin_edges[:-1], rel_freq, width=np.diff(bin_edges), align='edge',
           alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_name:
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    
    print(f"N = {len(data)}, μ = {np.mean(data):.2f}, σ = {np.std(data):.2f}")

def compute_cos_theta_action(file):
    with open(file, 'r') as f:
        data = json.load(f)

    theta_all_agents = {}

    for agent in data.keys():
        results = data[agent]
        theta_list = []

        for D_state, infos in results.items():
            config = list(infos["results_per_config"].values())[0]
            actions_D = config.get("action", [])
            # actions_D is a list of episodes, each episode is a list of actions
            for episode_actions in actions_D:
                print(len(episode_actions))
                episode_actions = np.array(episode_actions)
                if len(episode_actions) < 2:
                    continue
                for i in range(len(episode_actions) - 1):
                    a1 = episode_actions[i]
                    a2 = episode_actions[i + 1]
                    norm1 = np.linalg.norm(a1)
                    norm2 = np.linalg.norm(a2)
                    if norm1 == 0 or norm2 == 0:
                        continue
                    cos_theta = np.dot(a1, a2) / (norm1 * norm2)
                    theta_list.append(cos_theta)
        theta_all_agents[agent] = theta_list

    return theta_all_agents
    

if __name__ == "__main__":
    file = ['grid_search/66/result_evaluation_retina_.json','grid_search/65/result_evaluation_retina_.json']
    title = 'D_success_rate'
    plot_success_rate_D_state(file,title)    
    # cos_theta_all_agents = compute_cos_theta_action(file)
    # i=0
    # for agent,cos_theta_agent in cos_theta_all_agents.items():
    #     hist_scientific(cos_theta_agent,xlabel=r'$\cos(\theta)$',title='',save_name=f"fig/histogramme_agent_{i}_cos_theta_no_title",bins=20)
    #     i+=1
    
    