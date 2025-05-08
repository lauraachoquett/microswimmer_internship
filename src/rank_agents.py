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
from src.plot import analyze_and_visualize_agent_data

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


def merge_agent_stats(agent_stats_lists, agents_file):
    merged_stats = defaultdict(
        lambda: {"mean_reward": 0, "mean_reward_t": 0, "mean_reward_d": 0, "count": 0}
    )

    for agent_stats in agent_stats_lists:
        for agent in agent_stats:
            name = agent["agent_name"]
            if agent["agent_name"] in agents_file:
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


def rank_agents_all_criterion(files_results, agents_file):
    agent_stats_lists = []
    for results in files_results:
        with open(results, "r") as f:
            data = json.load(f)
        agent_stats = rank_agents_by_rewards(data, False)
        agent_stats_lists.append(agent_stats)
    merged_stats = merge_agent_stats(agent_stats_lists, agents_file)

    filtered_stats = [
        agent
        for agent in merged_stats
        if (agent["count"] >= 0 and agent["mean_reward"] > -50)
    ]

    filtered_stats = sorted(
        filtered_stats, key=lambda x: x["mean_reward"], reverse=True
    )
    print("\nMerged Top 10 agents by mean_reward :")
    for i, agent in enumerate(filtered_stats, 1):
        print(
            f"{i} Mean Reward: {agent['mean_reward']:.3f}__{agent['count']}__ __ {agent['agent_name']}__{agent['training type']} "
        )
        print()
    return filtered_stats







if __name__ == "__main__":
    types = ["ondulating", "curve_minus", "curve_plus", "line"]
    file = "results_evaluation"
    files_results = []
    for type in types:
        # files_results.extend(
        #     [
        #         f"results_evaluation/result_evaluation_east_05_{type}.json",
        #         f"results_evaluation/result_evaluation_west_05_{type}.json",
        #         f"results_evaluation/result_evaluation_north_05_{type}.json",
        #         f"results_evaluation/result_evaluation_south_05_{type}.json",
        #     ]
        # )
        files_results.extend(
            [
                f"results_evaluation/result_evaluation_rankine_a_05__cir_3_center_1_075_{type}.json"
            ]
        )
        # files_results.extend([f"results_evaluation/result_evaluation_free_{type}.json"])
    print("Overall ranking of agents:")
    print(files_results)
    agents_file = []

    directory_path = Path("agents/")

    for item in directory_path.iterdir():
        agents_file.append(os.path.join(directory_path, item.name))

    stats = rank_agents_all_criterion(files_results,agents_file)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    save_rank_file = os.path.join(file, f"results_rank_overall_rankine.json")
    with open(save_rank_file, "w") as f:
        json.dump(stats, f, indent=4)

    file_path = "results_evaluation/results_rank_overall_rankine.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    # analyze_and_visualize_agent_data(data=data,name_fig='result_3_agents')
    # file_path = "/Users/laura/Documents/MVA/Stage_Harvard/Project/results_evaluation/results_rank_overall_beta_values.json"
    # plot_return_beta(file_path)
