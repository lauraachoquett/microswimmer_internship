import json
import os
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns

colors_default = plt.cm.tab10.colors
from src.generate_path import (
    generate_curve,
    generate_demi_circle_path,
    generate_random_ondulating_path,
)
from src.simulation import rankine_vortex, uniform_velocity

def video_trajectory(
    fig,
    ax,
    trajectory,
    path,
    title,
    a=0,
    center=np.zeros(2),
    cir=0,
    dir=np.zeros(2),
    norm=0,
    plot_background=False,
    path_save_video ='',
    color_id=0,
    colors=plt.cm.tab10.colors,
    label="",
):
    trajectory = np.array(trajectory)
    step = max(1, len(trajectory) // 1000)  
    trajectory = trajectory[::step]
    print(trajectory.shape)
    # Initialisation de la ligne de trajectoire animée
    line, = ax.plot([], [], color=colors[color_id], linewidth=0.9)
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
        x, y = trajectory[:i + 1, 0], trajectory[:i + 1, 1]
        line.set_data(x, y)
        start_dot.set_offsets([trajectory[0]])
        end_dot.set_offsets([trajectory[i]])
        return line, start_dot, end_dot

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), init_func=init,
                                blit=True, interval=20)

    ani.save(path_save_video, writer='ffmpeg', dpi=200)

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
            """indices = np.linspace(0, len(path) - 1, list_state[1]).astype(int)
            path_sampled = path[indices]
            ax.plot(path_sampled[:, 0], path_sampled[:, 1], label='path', color='black', linewidth=2)
            """
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
        states = trajectories_list
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


if __name__ == "__main__":
    path_json_file = "results_evaluation/result_evaluation_retina.json"
    agent_file = "agents/agent_TD3_2025-04-18_13-33"
    plot_success_rate(path_json_file, agent_file)
