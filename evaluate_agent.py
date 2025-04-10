
from statistics import mean
from utils import random_bg_parameters
from sde import uniform_velocity,rankine_vortex,plot_background_velocity
import matplotlib.pyplot as plt
import numpy as np
import os
colors = plt.cm.tab10.colors
from generate_path import generate_curve

def evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig,file_name,random_parameters):
    rewards = []
    t_max = config['t_max']
    path = config['path']
    tree = config['tree']
    p_target = config['p_target']
    steps_per_action=config['steps_per_action']
    t_max =config['t_max']
    Dt_action = config['Dt_action']
    Dt_sim = Dt_action/steps_per_action
    p_0 = config['p_0']
    beta =config['beta']
    iter=0
    episode_num = 0
    episode_reward=0
    state,done = env.reset(tree,path),False
    states_episode = []
    states_list_per_episode=[]
    x=p_0
    u_bg = config['u_bg']
    D = config['D']
    type=''
    threshold = config['threshold']

    count_succes = 0

    if random_parameters : 
        dir, norm,center,a ,cir = random_bg_parameters()
    else : 
        dir, norm,center,a ,cir= np.zeros(2),0, np.zeros(2),0,0

    if u_bg.any() != 0:
        type='uniform'
        norm = np.linalg.norm(u_bg)
        dir = u_bg/norm
    
    if config['random_curve']:
        k = (np.random.rand()-0.5)*4
        path = generate_curve(p_0,p_target,k,config['nb_points_path'])

    while episode_num <eval_episodes:
        states_episode.append(x)
        iter+=1

        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        if config['uniform_bg']:
            u_bg = uniform_velocity(dir,norm)
            type='uniform'
        if config['rankine_bg']:
            u_bg = rankine_vortex(x,a,center,cir)
            type='rankine'
            
        next_state,x,reward,done,_ = env.step(action,tree,path,p_target,beta,D,u_bg,threshold)

        state = next_state
        episode_reward += reward
        rewards.append(episode_reward)
        
        if done or iter*Dt_sim> t_max: 
            if done : 
                count_succes+=1
            states_list_per_episode.append([np.array(states_episode),iter])
            state,done =env.reset(tree,path),False
            iter = 0
            episode_reward=0
            episode_num+=1
            x=p_0
            states_episode=[]


    path_save_fig = os.path.join(save_path_result_fig, file_name)

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, list_state in enumerate(states_list_per_episode[:3]):
        indices = np.linspace(0, len(path) - 1, list_state[1]).astype(int)
        path_sampled = path[indices]
        ax.plot(path_sampled[:, 0], path_sampled[:, 1], label='path', color='black', linewidth=2)
        states = list_state[0]
        ax.plot(states[:, 0], states[:, 1], color=colors[idx])
        ax.scatter(states[-1, 0], states[-1, 1], color=colors[idx])
    ax.set_aspect('equal')
    x_bound = ax.get_xlim()  # Retourne un tuple (x_min, x_max)
    y_bound = ax.get_ylim()  # Retourne un tuple (y_min, y_max)
    plot_background_velocity(type,x_bound,y_bound,a,center,cir,dir,norm)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if type == 'uniform':
        ax.set_title(f"Trajectories - norm : {norm}")
    if type=='rankine':
        ax.set_title(f"Trajectories - a : {a} - circulation : {cir}")
    fig.savefig(path_save_fig, dpi=100, bbox_inches='tight')

    plt.close(fig) 
            

    return mean(rewards),np.std(np.array(rewards)),count_succes/eval_episodes