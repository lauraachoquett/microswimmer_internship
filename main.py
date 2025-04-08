import numpy as np
import torch
import argparse
import os
from statistics import mean
import utils
import TD3
from datetime import datetime 
from env_swimmer import MicroSwimmer
from generate_path import *
import matplotlib.pyplot as plt
from invariant_state import *
from math import sqrt
import pickle
from sde import rankine_vortex,uniform_velocity,plot_background_velocity

colors = plt.cm.tab10.colors


def evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig,file_name,random):
    rewards = []
    t_max = config['t_max']
    path = config['path']
    p_target = config['p_target']
    steps_per_action=config['steps_per_action']
    n_t_sim = config['n_t_sim']
    t_init =config['t_init']
    t_max =config['t_max']
    Dt_sim = (t_max-t_init)/n_t_sim
    p_0 = config['p_0']
    iter=0
    episode_num = 0
    episode_reward=0
    state,done = env.reset(path),False
    states_episode = []
    states_list_per_episode=[]
    x=p_0
    u_bg = np.zeros(2)
    D = config['D']
    type=''

    count_succes = 0
    if random : 
        dir, norm,center,a ,cir = random_bg_parameters()
    else : 
        dir, norm,center,a ,cir= np.zeros(2),0, np.zeros(2),0,0
    
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
            
        next_state,x,reward,done,_ = env.step(action,path,p_target,D,u_bg)

        state = next_state
        episode_reward += reward
        rewards.append(episode_reward)
        
        if done or iter*Dt_sim> t_max: 
            if done : 
                count_succes+=1
            states_list_per_episode.append(np.array(states_episode))
            state,done =env.reset(path),False
            iter = 0
            episode_reward=0
            episode_num+=1
            x=p_0
            states_episode=[]


    plt.close()  
    path_save_fig = os.path.join(save_path_result_fig, file_name)

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, list_state in enumerate(states_list_per_episode[:3]):
        indices = np.linspace(0, len(path) - 1, n_t_sim).astype(int)
        path_sampled = path[indices]
        ax.plot(path_sampled[:, 0], path_sampled[:, 1], label='path', color='black', linewidth=2)
        ax.plot(list_state[:, 0], list_state[:, 1], color=colors[idx])
        ax.scatter(list_state[-1, 0], list_state[-1, 1], color=colors[idx])
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

def random_bg_parameters():
    dir = np.random.uniform(-1,1,2)
    dir = dir/np.linalg.norm(dir)
    norm = np.random.rand()*0.6

    a = np.random.rand()
    center = [np.random.rand()*2,np.random.rand()]
    cir = (np.random.rand()-0.5)*3
    return dir,norm,center,a,cir

def run_expe(config):
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"agent_TD3_{timestamp}"

    os.makedirs(file_name, exist_ok=True) 
    with open(os.path.join(file_name,'config.pkl'), "wb") as f:
        pickle.dump(config, f)


    path = config['path']
    p_target = config['p_target']
    x_0 = config['x_0']
    p_0 = config['p_0']

    print(f'Starting point : {p_0}')
    print(f'Target point : {path[-1]}')

    C = config['C']
    u_bg = config['u_bg']
    n_t_sim = config['n_t_sim']
    t_init =config['t_init']
    t_max =config['t_max']
    Dt_sim = (t_max-t_init)/n_t_sim
    print(f"Delta de temps entre chaque itération du solver : {Dt_sim}")

    steps_per_action=config['steps_per_action']
    threshold = config['threshold']
    Dt_action = Dt_sim*steps_per_action
    beta = config['beta']

    print(f"Delta de temps entre chaque action : {Dt_action}")

    env = MicroSwimmer(x_0,C,Dt_sim,threshold,beta)

    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    nb_episode = config['nb_episode']
    batch_size = config['batch_size']
    eval_freq = config['eval_freq']
    eval_episodes = config['eval_episodes']
    episode_start_update = config['episode_start_update']
    save_model = config['save_model']
    episode_update=config['episode_per_update']
    agent = TD3.TD3(state_dim,action_dim,max_action)

    replay_buffer = utils.ReplayBuffer(state_dim,action_dim)

    save_path_result = f"./{file_name}/results"
    save_path_result_fig = os.path.join(save_path_result,'fig/')
    if not os.path.exists(save_path_result):
        os.makedirs(save_path_result)
        os.makedirs(save_path_result_fig)
    save_path_model = f"./{file_name}/models/agent"
    if save_model and not os.path.exists(save_path_model):
        os.makedirs(save_path_model)

    iter = 0
    episode_reward = 0
    episode_num = 0
    training_reward=[]
    best_eval_result=-np.inf

    #random_eval = evaluate_agent(agent,env,eval_episodes,config,save_path_result)
    #print(f"Initial random evaluation : {random_eval}")
    state,done = env.reset(path),False
    print("Beginning of the training...")
    count_reach_target=0

    x=p_0
    u_bg = np.zeros(2)
    D = config['D']

    if config['random_curve']:
        k = 2
        path = generate_curve(p_0,p_target,k,config['nb_points_path'])

    ########### TRAINING LOOP ###########
    while episode_num <nb_episode:
        iter+=1

        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        if episode_num>100: 
            if config['uniform_bg']:
                u_bg = uniform_velocity(dir,norm)
            if config['rankine_bg']:
                u_bg = rankine_vortex(x,a,center,cir)

        next_state,x,reward,done,_ = env.step(action,path,p_target,D,u_bg)
        replay_buffer.add(state.flatten(), action, next_state.flatten(), reward, done)
            

        state = next_state
        episode_reward += reward

        if  episode_num%episode_update==0 and episode_num>episode_start_update:
            agent.train(replay_buffer, batch_size)
        if done:
            count_reach_target+=1
        if done or iter*Dt_sim> t_max: 
            state,done =env.reset(path),False
            training_reward.append(episode_reward)

            if episode_num%eval_freq==0:
                print(f"Total iter: {iter+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f} Counter target reached: {count_reach_target}")
                path_save_fig= os.path.join(save_path_result_fig,"training_reward.png")
                eval_rew = evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig,"eval_during_training",False)
                #evaluations.append(eval)
                print(f"Eval result : {eval_rew}")
                if best_eval_result<eval_rew : 
                    best_eval_result=eval_rew
                    if save_model: 
                        agent.save(save_path_model)
                        print("Best reward during evaluation : Model saved")
                plt.close()
                plt.plot(training_reward)
                plt.xlabel("episode")
                plt.ylabel("reward")
                plt.savefig(path_save_fig,dpi=100,bbox_inches='tight')

            iter = 0
            episode_reward=0
            episode_num+=1
            dir, norm,center,a ,cir=random_bg_parameters()
            if episode_num>100:
                if config['random_curve']:
                    k = (np.random.rand()-0.5)*4
                    path = generate_curve(p_0,p_target,k,config['nb_points_path'])

    agent.save(os.path.join(save_path_model,'last'))


def evaluate_after_training(agent_name,file_name_or,type,translation=None,theta=None):
    if translation is not None : 
        translation = translation
    else : 
        translation = np.zeros(2)

    if theta is not None : 
        theta = theta
        sin_th = sin(theta)
        cos_th = cos(theta)
        R = np.array([[cos_th, -sin_th],
                    [sin_th,  cos_th]])
    else : 
        R = np.eye(2)
        
    p_target = R@[4,4]+translation
    p_0 = R@np.zeros(2)+translation
    p_1 = [1/4,-1/4]+translation
    nb_points_path = 500
    if type =='two_line':
        path = generate_line_two_part(p_0,p_1,p_target,nb_points_path)
    if type =='circle' : 
        path = generate_demi_circle_path(p_0,p_target,nb_points_path)
    if type=='ondulating':
        path = generate_random_ondulating_path(p_0,p_target,nb_points_path,amplitude = 0.7,frequency=3)


    config_eval = {
        'x_0' : np.zeros(2),
        'C' : 1,
        'D' : 0.1,
        'u_bg' : np.array([0,1])*0.0,
        'threshold' : 0.02,
        't_max': 20,
        'n_t_sim':600,
        't_init':0,
        'steps_per_action':5,
        'nb_episode':800,
        'batch_size':256,
        'eval_freq':50,
        'save_model':True,
        'eval_episodes' : 100,
        'episode_start_update' : 10,
        'path' : path,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :100,
        'load_model':"",
        'episode_per_update' :3,
        'discount_factor' : 1,
        'beta':0.25,
        'uniform_bg':True,
        'rankine_bg':False,
        'random_curve' : False,
        'nb_points_path':500,
        'D_varying':True
    }

    policy_file = os.path.join(agent_name,'models/agent')
    n_t_sim = config_eval['n_t_sim']
    t_init =config_eval['t_init']
    t_max =config_eval['t_max']
    Dt_sim = (t_max-t_init)/n_t_sim
    env = MicroSwimmer(config_eval['x_0'],config_eval['C'],Dt_sim,config_eval['threshold'],config_eval['beta'])

    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    agent = TD3.TD3(state_dim,action_dim,max_action)

    save_path_eval = os.path.join(agent_name,'eval_bg/')
    os.makedirs(save_path_eval, exist_ok=True)
    agent.load(policy_file)


    nb_D=15
    mean_reward_D = np.zeros(nb_D)
    std_reward_D = np.zeros(nb_D)
    success_rate_D = np.zeros(nb_D)
    D_values = np.linspace(0.07, 0.5, nb_D)

    for idx, D in enumerate(D_values):
        config_eval['D'] = D
        file_name = file_name_or + f'_{idx}'
        mean_reward, std_reward, success_rate = evaluate_agent(agent, env, config_eval['eval_episodes'],
                                                                config_eval, save_path_eval,
                                                                file_name=file_name, random=False)
        mean_reward_D[idx] = mean_reward
        std_reward_D[idx] = std_reward
        success_rate_D[idx] = success_rate

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(D_values, mean_reward_D, label="Mean Reward")
    axs[0].fill_between(D_values,
                        mean_reward_D - std_reward_D,
                        mean_reward_D + std_reward_D,
                        alpha=0.3, color='blue', label="±1 Std")
    axs[0].set_title('Mean reward with respect to the noise')
    axs[0].set_xlabel('D')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    axs[1].plot(D_values, success_rate_D, color='green')
    axs[1].set_title('Success rate with respect to the noise')
    axs[1].set_xlabel('D')
    axs[1].set_ylabel('Success rate')

    fig.tight_layout()
    plt.show()


    
        

if __name__=='__main__':
    p_target = [2,0]
    p_0 = np.zeros(2)
    nb_points_path = 500
    path = generate_random_ondulating_path(p_0,p_target,nb_points_path)
    config = {
        'x_0' : np.zeros(2),
        'C' : 1,
        'D' : 0.1,
        'u_bg' : np.array([0,1])*0.0,
        'threshold' : 0.02,
        't_max': 5,
        'n_t_sim':200,
        't_init':0,
        'steps_per_action':5,
        'nb_episode':800,
        'batch_size':256,
        'eval_freq':50,
        'save_model':True,
        'eval_episodes' : 3,
        'episode_start_update' : 10,
        'path' : path,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :100,
        'load_model':"",
        'episode_per_update' :3,
        'discount_factor' : 1,
        'beta':0.25,
        'uniform_bg':False,
        'rankine_bg':True,
        'random_curve' : True,
        'nb_points_path':500,
    }
    #run_expe(config)
    ### EVAL ## 



    evaluate_after_training('agent_TD3_2025-04-08_15-07',file_name_or=f"eval_with_ondulating_path",type='ondulating')


