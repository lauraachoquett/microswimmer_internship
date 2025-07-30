import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Paramètres ===
agent_name = "agents/agent_TD3_2025-07-10_10-33"  # à remplacer ou automatiser
path_type = "hard_helix"  # ou "curve_minus", etc.
perturbations = [
    "free",
    "dir1_05",
    "dir2_05",
    "dir3_05",
    "dir4_05",
    "dir5_05",
    "dir6_05",
    "rankine_a_05_cir_0_5_2_pi_center_0_06_02"
]
labels = [
    "No Flow","dir1", "dir2", "dir3", "dir4","dir5","dir6", "Rankine vortex"
]
file_dir = "results_evaluation"
save_path_eval = os.path.join(agent_name, "eval_bg")
os.makedirs(save_path_eval, exist_ok=True)

# === Récupération des données ===
data_all = {
    "rewards": {},
    "rewards_time": {},
    "rewards_distance": {}
}

for perturb, label in zip(perturbations, labels):
    file_path = os.path.join(file_dir, f"result_evaluation_{perturb}_{path_type}.json")
    with open(file_path, "r") as f:
        results = json.load(f)

    agent_results = results[agent_name]
    data_all["rewards"][label] = agent_results["rewards"]
    data_all["rewards_time"][label] = agent_results["rewards_time"]
    data_all["rewards_distance"][label] = agent_results["rewards_distance"]

# === Préparation du plot ===
sns.set(style="whitegrid", font_scale=1.4)
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

titles = [
    r"$\mathrm{Total\ reward}\ J$",
    r"$\mathrm{Time\ reward}\ J_t$",
    r"$\mathrm{Distance\ reward}\ J_d$"
]
keys = ["rewards", "rewards_time", "rewards_distance"]
colors = sns.color_palette("Set2", len(perturbations))

for ax, key, title in zip(axs, keys, titles):
    data_plot = []
    perturb_labels = []

    for i, label in enumerate(labels):
        rewards = data_all[key][label]
        data_plot.extend(rewards)
        perturb_labels.extend([label] * len(rewards))

    sns.boxplot(x=perturb_labels, y=data_plot, hue=perturb_labels,
                ax=ax, palette=colors, legend=False, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel("Reward")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Sauvegarde
filename_base = f"{path_type}_perturbation_boxplot_eval"
plt.tight_layout()
plt.savefig(os.path.join(save_path_eval, f"{filename_base}.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_path_eval, f"{filename_base}.png"), bbox_inches="tight", dpi=400)
plt.close()

print(f"Figure sauvegardée dans : {os.path.join(save_path_eval, f'{filename_base}.pdf')}")