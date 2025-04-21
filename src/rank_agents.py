from statistics import mean
import json
from collections import defaultdict
import os 
from datetime import datetime 

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def rank_agents_by_rewards(results,print_stats=True):
    # Calculer les moyennes pour chaque agent
    agent_stats = []
    for agent_name, stats in results.items():
        if not 'agent' in agent_name:
            continue
        mean_reward = mean(stats['rewards'])
        mean_reward_t = mean(stats['rewards_time'])
        mean_reward_d = mean(stats['rewards_distance'])
        training_type = stats['training type']
        agent_stats.append({
            'agent_name': agent_name,
            'training type': training_type,
            'mean_reward': mean_reward,
            'mean_reward_t': mean_reward_t,
            'mean_reward_d': mean_reward_d
        })

    # Trier les agents par chaque critère
    sorted_by_reward = sorted(agent_stats, key=lambda x: x['mean_reward'], reverse=True)
    sorted_by_reward_t = sorted(agent_stats, key=lambda x: x['mean_reward_t'], reverse=True)
    sorted_by_reward_d = sorted(agent_stats, key=lambda x: x['mean_reward_d'], reverse=True)

    # Afficher les trois meilleurs agents pour chaque critère
    if print_stats:
        print("Top 5 agents by mean_reward:")
        for i, agent in enumerate(sorted_by_reward[:5], 1):
            print(f"{i} Mean Reward: {agent['mean_reward']:.3f} __ {agent['agent_name']} ")

        print("\nTop 5 agents by mean_reward_t:")
        for i, agent in enumerate(sorted_by_reward_t[:5], 1):
            print(f"{i} Mean Reward Time: {agent['mean_reward_t']:.3f} __ {agent['agent_name']}")

        print("\nTop 5 agents by mean_reward_d:")
        for i, agent in enumerate(sorted_by_reward_d[:5], 1):
            print(f"{i} Mean Reward Distance: {agent['mean_reward_d']:.3f} __ {agent['agent_name']}")

    return agent_stats
def merge_agent_stats(agent_stats_lists):
    merged_stats = defaultdict(lambda: {'mean_reward': 0, 'mean_reward_t': 0, 'mean_reward_d': 0, 'count': 0})

    for agent_stats in agent_stats_lists:
        for agent in agent_stats:
            name = agent['agent_name']
            if agent['training type']['random_curve'] == True:
                if '04-16' in agent['agent_name'] or '04-17' in agent['agent_name'] or '04-18' in agent['agent_name']:
                    merged_stats[name]['training type'] = agent['training type']
                    merged_stats[name]['mean_reward'] += agent['mean_reward']
                    merged_stats[name]['mean_reward_t'] += agent['mean_reward_t']
                    merged_stats[name]['mean_reward_d'] += agent['mean_reward_d']
                    merged_stats[name]['count'] += 1
            else:
                merged_stats[name]['training type'] = agent['training type']
                merged_stats[name]['mean_reward'] += agent['mean_reward']
                merged_stats[name]['mean_reward_t'] += agent['mean_reward_t']
                merged_stats[name]['mean_reward_d'] += agent['mean_reward_d']
                merged_stats[name]['count'] += 1

    final_stats = []
    for name, stats in merged_stats.items():
        final_stats.append({
            'agent_name': name,
            'training type': stats['training type'],
            'mean_reward': stats['mean_reward'] / stats['count'],
            'mean_reward_t': stats['mean_reward_t'] / stats['count'],
            'mean_reward_d': stats['mean_reward_d'] / stats['count'],
            'count': stats['count']
        })

    return final_stats

def rank_agents_all_criterion(files_results):
    agent_stats_lists = []
    for results in files_results:
        with open(results, 'r') as f:
            data = json.load(f)
        agent_stats = rank_agents_by_rewards(data, False)
        agent_stats_lists.append(agent_stats)
    merged_stats = merge_agent_stats(agent_stats_lists)
    
    filtered_stats = [agent for agent in merged_stats if (agent['count'] >= 20 and agent['mean_reward']>-50)]
    
    filtered_stats = sorted(filtered_stats, key=lambda x: x['mean_reward'], reverse=True)
    print("\nMerged Top 10 agents by mean_reward (with more than 20 evaluations):")
    for i, agent in enumerate(filtered_stats, 1):
        print(f"{i} Mean Reward: {agent['mean_reward']:.3f}__{agent['count']}__ __ {agent['agent_name']}__{agent['training type']} ")
        print()
    return filtered_stats

if __name__ == '__main__':
    types = ['ondulating','curve_minus','curve_plus','line','circle']
    file = "results_evaluation"
    files_results = []
    for type in types :
        files_results.extend([f'results_evaluation/result_evaluation_east_05_{type}.json', f'results_evaluation/result_evaluation_west_05_{type}.json', f'results_evaluation/result_evaluation_north_05_{type}.json',f'results_evaluation/result_evaluation_south_05_ondulating.json'])
        files_results.extend([f'results_evaluation/result_evaluation_rankine_a_05__cir_3_center_1_075_{type}.json'])
        files_results.extend([f'results_evaluation/result_evaluation_free_{type}.json'])
    print("Overall ranking of agents:")
    stats = rank_agents_all_criterion(files_results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    save_rank_file = os.path.join(file,f"results_rank_overall_{timestamp}.json")
    with open (save_rank_file,"w") as f:
        json.dump(stats,f, indent=4)
    
    
    
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    file_path = "results_evaluation/results_rank_overall_2025-04-18_15.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    df = pd.json_normalize(data)
    
    training_columns = [col for col in df.columns if col.startswith('training type.') and not col.endswith('load_model')]
    df['training_type_str'] = df[training_columns].apply(
        lambda row: ', '.join([f"{col.split('.')[-1]}={row[col]}" for col in training_columns]), axis=1
    )
    
    # Filter agents with random_curve = True
    random_curve_agents = df[df['training_type_str'].str.contains('random_curve=True', na=False)]
    agent_counts_file = './results_evaluation/agent_random_curve.json'
    os.makedirs(os.path.dirname(agent_counts_file), exist_ok=True)
    random_curve_agents.to_json(agent_counts_file, orient='records', indent=4)
    
    # Count the number of agents per training type
    agent_counts = df['training_type_str'].value_counts().reset_index()
    agent_counts.columns = ['training_type', 'agent_count']
    print("Number of agents per training type:")
    print(agent_counts)

    # Filter training types with more than 5 agents
    filtered_training_types = agent_counts[agent_counts['agent_count'] >= 4]['training_type']
    df = df[df['training_type_str'].isin(filtered_training_types)]
    
    # Save the agent counts to a file
    agent_counts_file = './results_evaluation/agent_counts.json'
    os.makedirs(os.path.dirname(agent_counts_file), exist_ok=True)
    agent_counts.to_json(agent_counts_file, orient='records', indent=4)
    print(f"Agent counts saved to {agent_counts_file}")
    
    grouped = df.groupby('training_type_str')
    
    stats = grouped.agg({
        "mean_reward": ["mean", "std"],
        "mean_reward_t": ["mean", "std"],
        "mean_reward_d": ["mean", "std"]
    }).reset_index()
    
    stats.columns = ['training_type', 'mean_reward_mean', 'mean_reward_std', 
                     'mean_reward_t_mean', 'mean_reward_t_std', 
                     'mean_reward_d_mean', 'mean_reward_d_std']
    
    x = np.arange(len(stats)) 
    width = 0.25 
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, stats['mean_reward_mean'], width, label='mean_reward', capsize=5)
    ax.bar(x, stats['mean_reward_t_mean'], width, label='mean_reward_t', capsize=5)
    ax.bar(x + width, stats['mean_reward_d_mean'], width, label='mean_reward_d', capsize=5)
    ax.set_xlabel("Training")
    ax.set_ylabel("Reward")
    ax.set_title("Mean and Standard Deviation of Rewards by Training Type")
    ax.set_xticks(x)
    ax.set_xticklabels(stats['training_type'], rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig('./fig/rank.png', dpi=400, bbox_inches='tight')
    
   