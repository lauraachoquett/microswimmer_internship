import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Chargement des agents groupés par n_lookahead ===
with open('data/list_n_lookahaed_08-12.json') as f:
    agent_groups = json.load(f)

path_type = "ondulating"
perturbations = [
    "free", "east_05", "north_05", "south_05", "west_05",
    "rankine_a_05_cir_0_75_2_pi_center_1_0"
]
labels = [
    "No Flow", "East", "North", "South","West", "Rankine vortex"
]
file_dir = "results_evaluation"
save_path_eval = os.path.join("comparison_nlookahead", "eval_bg")
os.makedirs(save_path_eval, exist_ok=True)

# === Données collectées ===
metric_key = "rewards_time"
data = {"Reward": [], "Perturbation": [], "n_lookahead": []}
summary_stats = {}

# === Collecte des données ===
for perturb, label in zip(perturbations, labels):
    summary_stats[label] = {}
    for group_name, agent_list in agent_groups.items():
        rewards = []

        for agent in agent_list:
            agent_name = agent
            file_path = os.path.join(file_dir, f"result_evaluation_{perturb}_{path_type}.json")
            try:
                with open(file_path, "r") as f:
                    results = json.load(f)
                agent_data = results[agent_name]
                rewards.extend(agent_data[metric_key])
            except Exception as e:
                print(f"⚠️ Erreur pour {agent_name} dans {perturb} : {e}")

        data["Reward"].extend(rewards)
        data["Perturbation"].extend([label] * len(rewards))
        data["n_lookahead"].extend([group_name] * len(rewards))

        # Moyenne et écart-type
        if rewards:
            mean = np.mean(rewards)
            std = np.std(rewards)
            summary_stats[label][group_name] = (mean, std)
        else:
            summary_stats[label][group_name] = (np.nan, np.nan)

# === Affichage console ===
print("\n--- PERFORMANCE COMPARISON by n_lookahead ---")
for label in labels:
    print(f"\n  Perturbation: {label}")
    for group_name in sorted(agent_groups.keys(), key=int):
        mean, std = summary_stats[label][group_name]
        if not np.isnan(mean):
            print(f"    n_lookahead = {group_name}: {mean:.2f} ± {std:.2f}")
        else:
            print(f"    n_lookahead = {group_name}: (no data)")

# === Plot académique ===
sns.set(style="whitegrid", font_scale=1.6, rc={"axes.labelsize": 18, "axes.titlesize": 18})
plt.figure(figsize=(14, 8))
palette = sns.color_palette("Set2", n_colors=len(agent_groups))

ax = sns.boxplot(
    x="Perturbation", y="Reward", hue="n_lookahead",
    data=data, palette=palette, showfliers=False
)
ax.set_xlabel("")
ax.set_ylabel(r"Time Reward $J_t$", fontsize=18)
ax.set_title(r"Performance Comparison by $n$", fontsize=20)
ax.tick_params(axis='x', rotation=30)

# Légende
ax.legend(title=r"$n$", title_fontsize=14, fontsize=12, loc="upper right")

# Sauvegarde
filename = f"{path_type}_nlookahead_few_n_rewards_comparison_{metric_key}_08-12"
plt.tight_layout()
plt.savefig(os.path.join(save_path_eval, f"{filename}.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(save_path_eval, f"{filename}.png"), bbox_inches="tight", dpi=400)
plt.close()

print(f"\n📁 Figure saved to: {os.path.join(save_path_eval, filename)}.pdf/.png")

# # === Nouvelle figure : Reward global par n_lookahead (agrégation toutes perturbations) ===
# plt.figure(figsize=(10, 6))
# sns.set(style="whitegrid", font_scale=1.6)

# ax = sns.boxplot(
#     x=r"$n$", y=r"$J$",
#     data=data, palette=palette, showfliers=False
# )
# ax.set_xlabel(r"$n$", fontsize=16)
# ax.set_ylabel(r"Total Reward $J$ (all perturbations)", fontsize=16)
# ax.set_title(r"Overall Reward by $n$", fontsize=20)

# # Sauvegarde
# filename_agg = f"{path_type}_nlookahead_few_n_aggregated_boxplot_{metric_key}"
# plt.tight_layout()
# plt.savefig(os.path.join(save_path_eval, f"{filename_agg}.pdf"), bbox_inches="tight")
# plt.savefig(os.path.join(save_path_eval, f"{filename_agg}.png"), bbox_inches="tight", dpi=400)
# plt.close()

# print(f"📁 Aggregated figure saved to: {os.path.join(save_path_eval, filename_agg)}.pdf/.png")