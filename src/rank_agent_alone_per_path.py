import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Paramètres ===
agent_name = "agents/agent_TD3_2025-04-18_13-33"  # à remplacer ou automatiser
types = ["ondulating", "curve_minus", "curve_plus", "line"]
labels=['S-curve','Convex curve','Concave curve','Straight line']
file_dir = "results_evaluation"
save_path_eval = os.path.join(agent_name, "eval_bg")
os.makedirs(save_path_eval, exist_ok=True)

# === Récupération des données ===
data_all = {
    "rewards": {},
    "rewards_time": {},
    "rewards_distance": {}
}

for type_name in types:
    file_path = os.path.join(file_dir, f"result_evaluation_free_{type_name}.json")
    with open(file_path, "r") as f:
        results = json.load(f)
    
    agent_results = results[agent_name]
    data_all["rewards"][type_name] = agent_results["rewards"]
    data_all["rewards_time"][type_name] = agent_results["rewards_time"]
    data_all["rewards_distance"][type_name] = agent_results["rewards_distance"]

# === Préparation du plot ===
sns.set(style="whitegrid", font_scale=1.4)
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
titles = [
    r"$\mathrm{Total\ reward}\ J$",
    r"$\mathrm{Time\ reward}\ J_t$",
    r"$\mathrm{Distance\ reward}\ J_d$"
]
keys = ["rewards", "rewards_time", "rewards_distance"]
colors = sns.color_palette("Set2", len(types))

for ax, key, title in zip(axs, keys, titles):
    data_plot = []
    type_labels = []

    for i, type_name in enumerate(types):
        rewards = data_all[key][type_name]
        data_plot.extend(rewards)
        type_labels.extend([labels[i]] * len(rewards))

    sns.boxplot(x=type_labels, y=data_plot, hue=type_labels,
                ax=ax, palette=colors, legend=False,showfliers=False)
    ax.set_title(title)
    ax.set_ylabel("Reward")

    # Inclinaison des étiquettes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Ajustements finaux
plt.tight_layout()
plt.savefig(os.path.join(save_path_eval, f"free_boxplot_eval.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_path_eval, f"free_boxplot_eval.png"), bbox_inches="tight",dpi=400)
plt.close()

print(f"Figure sauvegardée dans : {os.path.join(save_path_eval, f'free_boxplot_eval.pdf')}")