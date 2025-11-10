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


all_data = {"value": [], "Perturbation": [], "Group": []}
summary_stats = {}


for perturb, label in zip(perturbations, labels):
    for group_name, agent_list in agent_groups.items():
        group_success = []

        for agent in agent_list:
            file_path = os.path.join(file_dir, f"result_evaluation_{perturb}_{path_type}.json")
            with open(file_path, "r") as f:
                results = json.load(f)

            agent_data = results[agent]
            # success_rate est supposé en 0-1, convertir en % (0-100)
            success_pct = agent_data["success_rate"] * 100
            group_success.append(success_pct)

        all_data["value"].extend(group_success)
        all_data["Perturbation"].extend([label] * len(group_success))
        all_data["Group"].extend([group_name] * len(group_success))

        mean = np.mean(group_success)
        std = np.std(group_success)
        summary_stats.setdefault(label, {})[group_name] = {"mean": mean, "std": std}


improvement = {}
for label in labels:
    mean_vel = summary_stats[label]["vel"]["mean"]
    mean_no_vel = summary_stats[label]["no_vel"]["mean"]
    if mean_no_vel != 0:
        delta = 100 * (mean_vel - mean_no_vel) / mean_no_vel
    else:
        delta = float("inf")
    improvement[label] = delta


output_json = {
    "summary_stats": summary_stats,
    "improvement_percent": improvement
}
with open(os.path.join(save_path_eval, f"{path_type}_success_rate_comparison.json"), "w") as f:
    json.dump(output_json, f, indent=4)


sns.set(style="whitegrid", font_scale=1.4)
plt.figure(figsize=(10, 6))
palette = {"vel": "#66c2a5", "no_vel": "#fc8d62"}

sns.boxplot(
    x="Perturbation", y="value", hue="Group", data=all_data,
    showfliers=False, palette=palette
)
plt.ylabel("Success Rate (%)")
plt.xlabel("")
plt.title(f"Success Rate Comparison on {path_type.capitalize()} Path")
plt.xticks(rotation=45)
plt.legend(title="Group")

plt.tight_layout()
plt.savefig(os.path.join(save_path_eval, f"{path_type}_success_rate_comparison.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_path_eval, f"{path_type}_success_rate_comparison.png"), bbox_inches="tight", dpi=400)
plt.close()

print(f"\nSuccess rate comparison saved to JSON and plots in {save_path_eval}")