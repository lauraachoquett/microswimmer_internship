import json
import os
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def rank_agents_by_rewards(results, print_stats=True):
    # Calculer les moyennes pour chaque agent
    agent_stats = []
    for agent_name, stats in results.items():
        if not "agent" in agent_name:
            continue
        mean_reward = mean(stats["rewards"])
        mean_reward_t = mean(stats["rewards_time"])
        mean_reward_d = mean(stats["rewards_distance"])
        training_type = stats["training type"]
        agent_stats.append(
            {
                "agent_name": agent_name,
                "training type": training_type,
                "mean_reward": mean_reward,
                "mean_reward_t": mean_reward_t,
                "mean_reward_d": mean_reward_d,
            }
        )

    # Trier les agents par chaque critère
    sorted_by_reward = sorted(agent_stats, key=lambda x: x["mean_reward"], reverse=True)
    sorted_by_reward_t = sorted(
        agent_stats, key=lambda x: x["mean_reward_t"], reverse=True
    )
    sorted_by_reward_d = sorted(
        agent_stats, key=lambda x: x["mean_reward_d"], reverse=True
    )

    # Afficher les trois meilleurs agents pour chaque critère
    if print_stats:
        print("Top 5 agents by mean_reward:")
        for i, agent in enumerate(sorted_by_reward[:5], 1):
            print(
                f"{i} Mean Reward: {agent['mean_reward']:.3f} __ {agent['agent_name']} "
            )

        print("\nTop 5 agents by mean_reward_t:")
        for i, agent in enumerate(sorted_by_reward_t[:5], 1):
            print(
                f"{i} Mean Reward Time: {agent['mean_reward_t']:.3f} __ {agent['agent_name']}"
            )

        print("\nTop 5 agents by mean_reward_d:")
        for i, agent in enumerate(sorted_by_reward_d[:5], 1):
            print(
                f"{i} Mean Reward Distance: {agent['mean_reward_d']:.3f} __ {agent['agent_name']}"
            )

    return agent_stats


def merge_agent_stats(agent_stats_lists,agents_file):
    merged_stats = defaultdict(
        lambda: {"mean_reward": 0, "mean_reward_t": 0, "mean_reward_d": 0, "count": 0}
    )

    for agent_stats in agent_stats_lists:
        for agent in agent_stats:
            name = agent["agent_name"]
            if (
                agent['agent_name'] in agents_file
            ):
                merged_stats[name]["training type"] = agent["training type"]
                merged_stats[name]["mean_reward"] += agent["mean_reward"]
                merged_stats[name]["mean_reward_t"] += agent["mean_reward_t"]
                merged_stats[name]["mean_reward_d"] += agent["mean_reward_d"]
                merged_stats[name]["count"] += 1
        

    final_stats = []
    for name, stats in merged_stats.items():
        final_stats.append(
            {
                "agent_name": name,
                "training type": stats["training type"],
                "mean_reward": stats["mean_reward"] / stats["count"],
                "mean_reward_t": stats["mean_reward_t"] / stats["count"],
                "mean_reward_d": stats["mean_reward_d"] / stats["count"],
                "count": stats["count"],
            }
        )

    return final_stats


def rank_agents_all_criterion(files_results,agents_file):
    agent_stats_lists = []
    for results in files_results:
        with open(results, "r") as f:
            data = json.load(f)
        agent_stats = rank_agents_by_rewards(data, False)
        agent_stats_lists.append(agent_stats)
    merged_stats = merge_agent_stats(agent_stats_lists,agents_file)

    filtered_stats = [
        agent
        for agent in merged_stats
        if (agent["count"] >= 0 and agent["mean_reward"] > -50)
    ]

    filtered_stats = sorted(
        filtered_stats, key=lambda x: x["mean_reward"], reverse=True
    )
    print("\nMerged Top 10 agents by mean_reward (with more than 20 evaluations):")
    for i, agent in enumerate(filtered_stats, 1):
        print(
            f"{i} Mean Reward: {agent['mean_reward']:.3f}__{agent['count']}__ __ {agent['agent_name']}__{agent['training type']} "
        )
        print()
    return filtered_stats


def analyze_and_visualize_agent_data(data, output_dir="./results_evaluation", fig_dir="./fig"):
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

    filtered_training_types = agent_counts[agent_counts["agent_count"] >= 5][
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
    fig_path = os.path.join(fig_dir, "return_per_training.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

def plot_return_beta(file_path):
    
    with open(file_path, 'r') as file:
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
    plt.scatter(betas, mean_rewards, label= "Overall mean return", marker='o', color=palette[0])
    plt.scatter(betas, mean_rewards_t, label=r"Time return: $-C \sum \Delta t_{sim}$", marker='o', color=palette[1])
    plt.scatter(betas, mean_rewards_d, label=r"Distance return: $-\beta \sum d$", marker='o', color=palette[2])
    
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
    plt.savefig("fig/rank_beta_return.png", dpi=200, bbox_inches='tight') 
if __name__ == "__main__":
    # types = ["ondulating", "curve_minus", "curve_plus", "line", "circle"]
    # file = "results_evaluation"
    # files_results = []
    # for type in types:
    #     files_results.extend(
    #         [
    #             f"results_evaluation/result_evaluation_east_05_{type}.json",
    #             f"results_evaluation/result_evaluation_west_05_{type}.json",
    #             f"results_evaluation/result_evaluation_north_05_{type}.json",
    #             f"results_evaluation/result_evaluation_south_05_ondulating.json",
    #         ]
    #     )
    #     files_results.extend(
    #         [
    #             f"results_evaluation/result_evaluation_rankine_a_05__cir_3_center_1_075_{type}.json"
    #         ]
    #     )
    #     files_results.extend([f"results_evaluation/result_evaluation_free_{type}.json"])
    # print("Overall ranking of agents:")
    
    # agents_file = []

    # directory_path = Path("agents/")

    # for item in directory_path.iterdir():
    #     if item.is_dir() and "agent_TD3" in item.name :
    #         if '2025-04-23' in item.name or '2025-04-22' in item.name:
    #             agents_file.append(os.path.join(directory_path, item.name))
    
    # stats = rank_agents_all_criterion(files_results,agents_file)
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    # save_rank_file = os.path.join(file, f"results_rank_overall_beta_values.json")
    # with open(save_rank_file, "w") as f:
    #     json.dump(stats, f, indent=4)



    # file_path = "results_evaluation/results_rank_overall_2025-04-18_15.json"
    # with open(file_path, "r") as f:
    #     data = json.load(f)
        
    # analyze_and_visualize_agent_data(data)
    file_path = "/Users/laura/Documents/MVA/Stage_Harvard/Project/results_evaluation/results_rank_overall_beta_values.json"
    plot_return_beta(file_path)
