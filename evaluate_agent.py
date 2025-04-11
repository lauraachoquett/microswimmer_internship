
from statistics import mean
from utils import random_bg_parameters
from sde import uniform_velocity,rankine_vortex
import matplotlib.pyplot as plt
import numpy as np
import os
colors = plt.cm.tab10.colors
from generate_path import generate_curve
from plot import plot_trajectories

def evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig,file_name,random_parameters,title='',plot=True,parameters=[],plot_background=False):
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
    u_bg = config['u_bg']
    D = config['D']
    type=''
    threshold = config['threshold']
    x = config['x_0']
    count_succes = 0

    if random_parameters : 
        dir, norm,center,a ,cir = random_bg_parameters()
    else : 
        if len(parameters)!=0:
            plot_background=True
            if config['uniform_bg']:
                dir,norm= parameters
                type='uniform'
                center=np.zeros(2),0,0 
            if config['rankine_bg']:
                type='rankine'
                center,a,cir = parameters
                dir,norm = np.zeros(2),0
        else: 
            dir, norm,center,a ,cir= np.zeros(2),0, np.zeros(2),0,0

    if u_bg.any() != 0:
        type='uniform'
        norm = np.linalg.norm(u_bg)
        dir = u_bg/norm
        plot_background=True
    
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
        if config['rankine_bg']:
            u_bg = rankine_vortex(x,a,center,cir)
            
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

    if plot : 
        path_save_fig = os.path.join(save_path_result_fig, file_name)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(path[:, 0], path[:, 1], label='path', color='black', linewidth=2,zorder=10)
        plot_trajectories(ax,states_list_per_episode[-3:],path,title,a,center,cir,dir,norm,plot_background,type=type)

        ax.set_aspect("equal")
        fig.savefig(path_save_fig, dpi=100, bbox_inches='tight')

        plt.close(fig) 

    return mean(rewards),np.std(np.array(rewards)),count_succes/eval_episodes,states_list_per_episode



