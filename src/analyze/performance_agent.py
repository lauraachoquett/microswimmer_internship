import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
import pandas as pd

def performance_stats_plot(file_path,file_path_free,agent_name,type,path,label):
    # === Paramètres ===
    save_path= os.path.join(agent_name, "eval_bg/performance/")

    os.makedirs(save_path, exist_ok=True)

    


    # === Chargement des données ===
    with open(file_path, 'r') as f:
        results = json.load(f)
        
    with open(file_path_free, 'r') as f:
        results_free = json.load(f)

    plot_t_l_d_per_episode = results[agent_name]["plot_t_l_d"]
    plot_t_l_d_per_episode_free = results_free['agents/agent_TD3_2025-04-18_13-33']["plot_t_l_d"]

    length_path = results[agent_name]["length_path"]
    U = 1
    threshold = 0.07
    t_star = (length_path-threshold)/ U

    # === Style ===
    sns.set(style="whitegrid", font_scale=1.4)

    # === Création grille temporelle commune ===
    T_max = max(
        ep[-1][0] for ep in plot_t_l_d_per_episode)/ t_star
    T_max_free = max(ep[-1][0] for ep in plot_t_l_d_per_episode_free)/ t_star
    t_common = np.linspace(0, T_max, 200)
    t_common_free = np.linspace(0, T_max_free, 200)

    # === Plot ===
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'hspace': 0.05})
    fig.suptitle(f"Policy Performance : {label}", fontsize=15)

    def compute_mean_std(plot_data,free=False):
        arr_target, arr_path = [], []
        for ep in plot_data:
            ep = np.array(ep)
            t = ep[:, 0] / t_star
            d_target = ep[:, 1] / threshold
            d_path = ep[:, 2] / threshold

            f_target = interp1d(t, d_target, kind='linear', bounds_error=False,
                                fill_value=(d_target[0], 0.0))
            f_path = interp1d(t, d_path, kind='linear', bounds_error=False,
                            fill_value=(d_path[0], 0.0))
            # axs[0].plot(t_common, f_target(t_common), color="#8cbed8",alpha=0.3)
            # axs[1].plot(t_common, f_path(t_common), color="#e3a010",alpha=0.2)

            if free :
                arr_target.append(f_target(t_common_free))
                arr_path.append(f_path(t_common_free))
            else : 
                arr_target.append(f_target(t_common))
                arr_path.append(f_path(t_common))

        arr_target = np.array(arr_target)
        arr_path = np.array(arr_path)
        
        return np.mean(arr_target, axis=0), np.min(arr_target, axis=0),np.max(arr_target, axis=0), \
            np.mean(arr_path, axis=0), np.min(arr_path, axis=0),np.max(arr_path, axis=0)

    # === Calcul des moyennes/std pour les deux ensembles ===
    mean_target, min_target, max_target,mean_path, min_path,max_path= compute_mean_std(plot_t_l_d_per_episode)
    mean_target_free, min_target_free,max_target_free, mean_path_free,min_path_free,max_path_free = compute_mean_std(plot_t_l_d_per_episode_free,True)



    # --- Distance to target
    axs[0].plot(t_common, mean_target, color="#1f77b4", label="Agent",linewidth=2)
    axs[0].fill_between(t_common,
                        min_target,
                        max_target,
                        color="#1f77b4", alpha=0.3)
    axs[0].plot(t_common_free, mean_target_free, color="black", linestyle="--", label="Reference")

    axs[0].set_ylabel(r"$l$ /  $\delta$")

    # --- Distance to path
    axs[1].plot(t_common, mean_path, color="#d62728", label="Agent")
    axs[1].fill_between(t_common,
                        min_path,
                    max_path,
                        color="#d62728", alpha=0.3)
    axs[1].plot(t_common_free, mean_path_free, color="black", linestyle="--", label="Reference")

    axs[1].set_ylabel(r"$d$ / $\delta$")
    axs[1].set_xlabel(r"$t$ / $t^*$")

    for ax in axs:
        ax.grid(True)
    name_fig = f"distances_vs_time_with_reference_min_max_{type}_{path}"
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, name_fig+'.png'),bbox_inches="tight", dpi=400)
    plt.savefig(os.path.join(save_path, name_fig+'.pdf'),bbox_inches="tight")
    plt.close()

    print("✅ figure with reference saved to:", os.path.join(save_path, name_fig))
    
    # === Statistiques supplémentaires pour le tableau JSON global ===

    summary_path = os.path.join(agent_name,"performance_summary.json")
    try:
        with open(summary_path, "r") as f:
            performance_summary = json.load(f)
    except FileNotFoundError:
        performance_summary = {}

    # Initialiser le niveau du path et type
    if path not in performance_summary:
        performance_summary[path] = {}
    if type not in performance_summary[path]:
        performance_summary[path][type] = {}

    # Calcul du T_max / t* par épisode
    t_max_list = [ep[-1][0] / t_star for ep in plot_t_l_d_per_episode]
    mean_T_max_over_t_star = float(np.mean(t_max_list))

    # Moyenne de d/threshold par épisode (pas interpolée !)
    mean_d_over_threshold_per_episode = []
    for ep in plot_t_l_d_per_episode:
        ep = np.array(ep)
        d_path = ep[:, 2] / threshold
        mean_d = np.mean(d_path)
        mean_d_over_threshold_per_episode.append(mean_d)

    mean_mean_d_over_threshold = float(np.mean(mean_d_over_threshold_per_episode))

    # Sauvegarde
    performance_summary[path][type]["mean_T_max_over_t_star"] = mean_T_max_over_t_star
    performance_summary[path][type]["t_max_list"] = t_max_list
    performance_summary[path][type]["mean_mean_d_over_threshold"] = mean_mean_d_over_threshold
    performance_summary[path][type]["mean_d_over_threshold_per_episode"] = mean_d_over_threshold_per_episode

    with open(summary_path, "w") as f:
        json.dump(performance_summary, f, indent=4)

    print("📊 Summary JSON updated at:", summary_path)
    
def box_plot_data_free(summary_path, agent_name):
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # === Chemins et styles ===
    output_dir = os.path.join(agent_name, 'eval_bg')
    os.makedirs(output_dir, exist_ok=True)

    types = ["line", "curve_minus", "curve_plus", "ondulating"]
    labels = ['Straight line', 'Convex curve', 'Concave curve', 'S-curve']
    colors = sns.color_palette("Set2", len(types))

    # === Chargement des données ===
    with open(summary_path, "r") as f:
        performance_summary = json.load(f)

    # === Extraction des données (1 ligne par épisode) ===
    data_tmax = []
    data_dmean = []

    for type_name, label in zip(types, labels):
        if type_name in performance_summary and "free" in performance_summary[type_name]:
            tmax_list = performance_summary[type_name]["free"].get("t_max_list", [])
            dmean_list = performance_summary[type_name]["free"].get("mean_d_over_threshold_per_episode", [])

            data_tmax.extend([{"Path": label, "Value": float(t)} for t in tmax_list])
            data_dmean.extend([{"Path": label, "Value": float(d)} for d in dmean_list])

    df_tmax = pd.DataFrame(data_tmax)
    df_dmean = pd.DataFrame(data_dmean)

    # === Style global ===
    sns.set(style="whitegrid", font_scale=1.4)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # === Boxplot Tmax / t* ===
    sns.boxplot(data=df_tmax, x="Path", y="Value", hue="Path",
                ax=axs[0], palette=colors, legend=False, showfliers=False)
    axs[0].set_ylabel(r"$T_{end} / t^*$")
    axs[0].set_xlabel("")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')

    # === Boxplot mean(d) / δ ===
    sns.boxplot(data=df_dmean, x="Path", y="Value", hue="Path",
                ax=axs[1], palette=colors, legend=False, showfliers=False)
    axs[1].set_ylabel(r"$\overline{d} / \delta$")
    axs[1].set_xlabel("")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')

    # === Sauvegarde ===
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"free_boxplot_time_dmean.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"free_boxplot_time_dmean.png"), bbox_inches="tight", dpi=400)
    plt.close()
    print(f"📦 Figure sauvegardée dans : {os.path.join(output_dir, 'free_boxplot_time_dmean.pdf')}")
    
def box_plot_data(summary_path, agent_name):
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # === Chemins et styles ===
    output_dir = os.path.join(agent_name, 'eval_bg')
    os.makedirs(output_dir, exist_ok=True)

    path_type = "ondulating"  # ou "curve_minus", etc.
    perturbations = [
        "free",
        "east_05",
        "south_05",
        "north_05",
        "west_05",
        "rankine_a_05_cir_0_75_2_pi_center_1_0"
    ]
    labels = [
        "No Flow", "East", "South", "North","West", "Rankine vortex"
    ]
    colors = sns.color_palette("Set2", len(perturbations))

    # === Chargement des données ===
    with open(summary_path, "r") as f:
        performance_summary = json.load(f)

    # === Extraction des données (1 ligne par épisode) ===
    data_tmax = []
    data_dmean = []

    for perturbations, label in zip(perturbations, labels):
        if path_type in performance_summary :
            tmax_list = performance_summary[path_type][perturbations].get("t_max_list", [])
            dmean_list = performance_summary[path_type][perturbations].get("mean_d_over_threshold_per_episode", [])

            data_tmax.extend([{"Perturbation": label, "Value": float(t)} for t in tmax_list])
            data_dmean.extend([{"Perturbation": label, "Value": float(d)} for d in dmean_list])

    df_tmax = pd.DataFrame(data_tmax)
    df_dmean = pd.DataFrame(data_dmean)

    # === Style global ===
    sns.set(style="whitegrid", font_scale=1.4)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # === Boxplot Tmax / t* ===
    sns.boxplot(data=df_tmax, x="Perturbation", y="Value", hue="Perturbation",
                ax=axs[0], palette=colors, legend=False, showfliers=False)
    axs[0].set_ylabel(r"$T_{end} / t^*$")
    axs[0].set_xlabel("")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')

    # === Boxplot mean(d) / δ ===
    sns.boxplot(data=df_dmean, x="Perturbation", y="Value", hue="Perturbation",
                ax=axs[1], palette=colors, legend=False, showfliers=False)
    axs[1].set_ylabel(r"$\overline{d} / \delta$")
    axs[1].set_xlabel("")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')

    # === Sauvegarde ===
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{path_type}_boxplot_time_dmean.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{path_type}_boxplot_time_dmean.png"), bbox_inches="tight", dpi=400)
    plt.close()
    print(f"📦 Figure sauvegardée dans : {os.path.join(output_dir, '{path_type}_boxplot_time_dmean.pdf')}")
    
if __name__ =='__main__':
    path = 'ondulating'
    type='free'
    file_path = f"results_evaluation/result_evaluation_{type}_{path}.json"
    file_path_free = f"results_evaluation/result_evaluation_free_D_0_{path}.json"
    agent_name = "agents/agent_TD3_2025-04-18_13-33"
    performance_stats_plot(file_path=file_path,file_path_free=file_path_free,agent_name=agent_name,type=type,path=path)
