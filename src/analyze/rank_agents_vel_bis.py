import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



with open('data/list_velocity.json') as f:
    agent_groups = json.load(f)
           


path_type = "ondulating"
perturbations = [
    "free", "west_05", "east_05", "north_05", "south_05",
    "rankine_a_05_cir_0_75_2_pi_center_1_0"
]
labels = [
    "Free", "West", "East", "North", "South", "Rankine vortex"
]
file_dir = "results_evaluation"
save_path_eval = os.path.join("comparison_vel_no_vel", "eval_bg")
os.makedirs(save_path_eval, exist_ok=True)


metrics_keys = {
    "J": "rewards",
    "J_t": "rewards_time",
    "J_d": "rewards_distance"
}
all_data = {k: {"value": [], "Perturbation": [], "Group": []} for k in metrics_keys}
summary_stats = {k: {} for k in metrics_keys}

for perturb, label in zip(perturbations, labels):
    for group_name, agent_list in agent_groups.items():
        group_rewards = {k: [] for k in metrics_keys}

        for agent in agent_list:
            file_path = os.path.join(file_dir, f"result_evaluation_{perturb}_{path_type}.json")
            with open(file_path, "r") as f:
                results = json.load(f)

            agent_data = results[agent]
            for k, key_json in metrics_keys.items():
                group_rewards[k].extend(agent_data[key_json])

        for k in metrics_keys:
            all_data[k]["value"].extend(group_rewards[k])
            all_data[k]["Perturbation"].extend([label] * len(group_rewards[k]))
            all_data[k]["Group"].extend([group_name] * len(group_rewards[k]))

            mean = np.mean(group_rewards[k])
            std = np.std(group_rewards[k])
            summary_stats[k].setdefault(label, {})[group_name] = (mean, std)
            
print("\n--- PERFORMANCE COMPARISON ---")
for k in metrics_keys:
    print(f"\nMetric: {k}")
    for label in labels:
        mean_vel, std_vel = summary_stats[k][label]["vel"]
        mean_novel, std_novel = summary_stats[k][label]["no_vel"]
        if mean_novel != 0:
            delta = 100 * (mean_vel - mean_novel) / abs(mean_novel)
        else:
            delta = float("inf")
        print(f"{label}: vel = {mean_vel:.2f} ± {std_vel:.2f} | no_vel = {mean_novel:.2f} ± {std_novel:.2f} → Δ = {delta:.1f}%")

sns.set(style="whitegrid", font_scale=1.4)
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
titles = [r"$\mathrm{Total\ reward}\ J$", r"$\mathrm{Time\ reward}\ J_t$", r"$\mathrm{Distance\ reward}\ J_d$"]
palette = {"vel": "#66c2a5", "no_vel": "#fc8d62"}

for ax, (k, data), title in zip(axs, all_data.items(), titles):
    sns.boxplot(
        x="Perturbation", y="value", hue="Group", data=data,
        ax=ax, showfliers=False, palette=palette
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Reward")
    ax.tick_params(axis='x', rotation=45)

axs[0].legend(title="Group")
axs[1].legend().set_visible(False)
axs[2].legend().set_visible(False)

filename = f"{path_type}_vel_vs_no_vel_boxplot"
plt.tight_layout()
plt.savefig(os.path.join(save_path_eval, f"{filename}.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_path_eval, f"{filename}.png"), bbox_inches="tight", dpi=400)
plt.close()

print(f"\n📁 Figures saved to: {os.path.join(save_path_eval, filename)}.pdf/.png")