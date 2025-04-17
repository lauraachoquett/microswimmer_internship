
from statistics import mean
from utils import random_bg_parameters
from sde import uniform_velocity,rankine_vortex
import matplotlib.pyplot as plt
import numpy as np
import os
colors = plt.cm.tab10.colors
from generate_path import generate_curve
from plot import plot_trajectories
import copy
def evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig,file_name,random_parameters,list_of_path_tree=None,title='',plot=True,parameters=[],plot_background=False):
    config = copy.deepcopy(config)
    parameters = copy.deepcopy(parameters)
    rewards_per_episode = []
    t_max = config['t_max']
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
    episode_rew_t=0
    episode_rew_d=0
    rewards_t_per_episode = []
    rewards_d_per_episode = []
    
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
            if config['uniform_bg']:
                dir,norm= parameters
                dir = np.array(dir)
                type='uniform'
                center,a,cir=np.zeros(2),0,0 
            if config['rankine_bg']:
                type='rankine'
                center,a,cir = parameters
                dir,norm = np.zeros(2),0
        else: 
            dir, norm,center,a ,cir= np.zeros(2),0, np.zeros(2),0,0

    if u_bg.any() != 0:
        type='uniform'
        norm = np.linalg.norm(u_bg)
        dir = np.array(u_bg/norm)
        plot_background=True
    

    if list_of_path_tree is not None : 
        path,tree = list_of_path_tree[0]
        nb_of_path = len(list_of_path_tree)
    else : 
        list_of_path_tree = [[config['path'],config['tree']]]
        path,tree = list_of_path_tree[0]
        nb_of_path=1
        
    state,done = env.reset(tree,path),False
    while episode_num <eval_episodes:
        states_episode.append(x)
        iter+=1

        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        if config['uniform_bg']:
            u_bg = uniform_velocity(np.array(dir),norm)
        if config['rankine_bg']:
            u_bg = rankine_vortex(x,a,center,cir)

        next_state,reward,done,info = env.step(action,tree,path,p_target,beta,D,u_bg,threshold)
        
        x= info['x']
        episode_rew_t += info['rew_t']
        episode_rew_d += info['rew_d']
        episode_reward += reward
        state = next_state
        
        if done or iter*Dt_sim> t_max: 
            if done : 
                count_succes+=1
            states_list_per_episode.append([np.array(states_episode),iter])
            state,done =env.reset(tree,path),False
            iter = 0
            episode_num+=1
            x=p_0
            states_episode=[]
            rewards_per_episode.append(episode_reward)
            rewards_t_per_episode.append(episode_rew_t)
            rewards_d_per_episode.append(episode_rew_d)
            episode_reward=0
            episode_rew_t=0
            episode_rew_d=0
            path,tree = list_of_path_tree[episode_num%nb_of_path]
        
            
            


    if plot : 
        path_save_fig = os.path.join(save_path_result_fig, file_name)
        fig, ax = plt.subplots(figsize=(10, 8))
        for elt in list_of_path_tree:
            path,_ = elt
            ax.plot(path[:, 0], path[:, 1], label='path', color='black', linewidth=2,zorder=10)
        ylim = ax.get_ylim()
        if ylim[1]-ylim[0]<1/3:
            ax.set_ylim(top=1.0,bottom=-1)
        plot_trajectories(ax,states_list_per_episode[-4:],path,title,a,center,cir,dir,norm,plot_background,type=type)
        ax.set_aspect("equal")
        fig.savefig(path_save_fig, dpi=100, bbox_inches='tight')

        plt.close(fig) 
        
    
    return rewards_per_episode, rewards_t_per_episode, rewards_d_per_episode,count_succes/eval_episodes,states_list_per_episode



