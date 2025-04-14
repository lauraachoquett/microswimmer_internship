from statistics import mean
def rank_agents_by_rewards(results):
    # Calculer les moyennes pour chaque agent
    agent_stats = []
    for agent_name, stats in results.items():
        if not 'agent' in agent_name:
            continue
        mean_reward = mean(stats['rewards'])
        mean_reward_t = mean(stats['rewards_time'])
        mean_reward_d = mean(stats['rewards_distance'])
        agent_stats.append({
            'agent_name': agent_name,
            'mean_reward': mean_reward,
            'mean_reward_t': mean_reward_t,
            'mean_reward_d': mean_reward_d
        })

    # Trier les agents par chaque critère
    sorted_by_reward = sorted(agent_stats, key=lambda x: x['mean_reward'], reverse=True)
    sorted_by_reward_t = sorted(agent_stats, key=lambda x: x['mean_reward_t'], reverse=True)
    sorted_by_reward_d = sorted(agent_stats, key=lambda x: x['mean_reward_d'], reverse=True)

    # Afficher les trois meilleurs agents pour chaque critère
    print("Top 5 agents by mean_reward:")
    for i, agent in enumerate(sorted_by_reward[:5], 1):
        print(f"{i} Mean Reward: {agent['mean_reward']:.3f} __ {agent['agent_name']} ")

    print("\nTop 5 agents by mean_reward_t:")
    for i, agent in enumerate(sorted_by_reward_t[:5], 1):
        print(f"{i} Mean Reward Time: {agent['mean_reward_t']:.3f} __ {agent['agent_name']}")

    print("\nTop 5 agents by mean_reward_d:")
    for i, agent in enumerate(sorted_by_reward_d[:5], 1):
        print(f"{i} Mean Reward Distance: {agent['mean_reward_d']:.3f} __ {agent['agent_name']}")

    return sorted_by_reward, sorted_by_reward_t, sorted_by_reward_d