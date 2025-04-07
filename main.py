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

colors = plt.cm.tab10.colors


def evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig):
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
    T_0 = (path[1]- path[0])/(np.linalg.norm(path[1]- path[0]))
    iter=0
    episode_num = 0
    episode_reward=0
    state,done = env.reset(path),False
    states_episode = []
    states_list_per_episode=[]
    x=p_0
    while episode_num <eval_episodes:
        states_episode.append(x)
        iter+=1
        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        next_state,x,reward,done,_ = env.step(action,path,p_target)
        state = next_state
        episode_reward += reward
        rewards.append(episode_reward)
        
        if done or iter*Dt_sim> t_max: 
            states_list_per_episode.append(np.array(states_episode))
            state,done =env.reset(path),False
            iter = 0
            episode_reward=0
            episode_num+=1
            x=p_0
            states_episode=[]

    plt.close()  
    path_save_fig = os.path.join(save_path_result_fig, "training_eval_trajectories.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, list_state in enumerate(states_list_per_episode):
        indices = np.linspace(0, len(path) - 1, n_t_sim).astype(int)
        path_sampled = path[indices]
        ax.plot(path_sampled[:, 0], path_sampled[:, 1], label='path', color='black', linewidth=2)
        ax.plot(list_state[:, 0], list_state[:, 1], color=colors[idx])
        ax.scatter(list_state[-1, 0], list_state[-1, 1], color=colors[idx])
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectories")

    fig.savefig(path_save_fig, dpi=100, bbox_inches='tight')
    plt.close(fig) 
            

    return mean(rewards)

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
    T_0 = (path[1]- path[0])/(np.linalg.norm(path[1]- path[0]))

    print(f'Starting point : {p_0}')
    print(f'Target point : {path[-1]}')

    C = config['C']

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

    random_eval = evaluate_agent(agent,env,eval_episodes,config,save_path_result)
    evaluations = [random_eval]
    print(f"Initial random evaluation : {random_eval}")
    state,done = env.reset(path),False
    print("Beginning of the training...")
    count_reach_target=0

    x=p_0

    ########### TRAINING LOOP ###########
    
    while episode_num <nb_episode:
        iter+=1

        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        next_state,x,reward,done,_ = env.step(action,path,p_target)
        replay_buffer.add(state.flatten(), action, next_state.flatten(), reward, done)
            

        state = next_state
        episode_reward += reward

        if  episode_num%episode_update==0 and episode_num>10:
            agent.train(replay_buffer, batch_size)
        if done:
            count_reach_target+=1
        if done or iter*Dt_sim> t_max: 
            state,done =env.reset(path),False
            training_reward.append(episode_reward)

            if episode_num%eval_freq==0:
                print(f"Total iter: {iter+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f} Counter target reached: {count_reach_target}")
                path_save_fig= os.path.join(save_path_result_fig,"training_reward.png")
                eval = evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig)
                #evaluations.append(eval)
                print(f"Eval result : {eval}")
                #np.save(save_path_result, evaluations)
                if save_model: 
                    agent.save(save_path_model)
                plt.close()
                plt.plot(training_reward)
                plt.xlabel("episode")
                plt.ylabel("reward")
                plt.savefig(path_save_fig,dpi=100,bbox_inches='tight')

            iter = 0
            episode_reward=0
            episode_num+=1

def evaluate_after_training(agent_name):
    translation = np.ones(2)*5
    theta = np.pi/8
    sin_th = sin(theta)
    cos_th = cos(theta)
    R = np.array([[cos_th, -sin_th],
                  [sin_th,  cos_th]])
    

    p_target = R@[2,0]+translation
    p_0 = R@np.zeros(2)+translation
    p_1 = [1/2,1]+translation
    nb_points_path = 500
    path = generate_demi_circle_path(p_0,p_target,nb_points_path)


    config_eval = {
        'x_0' : p_0,
        'C' : 1,
        'D' : 0.1,
        'threshold' : 0.02,
        't_max': 5,
        'n_t_sim':200,
        't_init':0,
        'steps_per_action':5,
        'nb_episode':500,
        'batch_size':256,
        'eval_freq':100,
        'save_model':True,
        'eval_episodes' : 3,
        'path' : path,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :50,
        'load_model':"",
        'episode_per_update' :3,
        'discount_factor' : 1,
        'beta':2
    }
    
    policy_file = os.path.join(agent_name,'models/agent')
    n_t_sim = config_eval['n_t_sim']
    t_init =config_eval['t_init']
    t_max =config_eval['t_max']
    Dt_sim = (t_max-t_init)/n_t_sim
    env = MicroSwimmer(config_eval['x_0'],config_eval['C'],Dt_sim,config_eval['threshold'],config['beta'])

    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    agent = TD3.TD3(state_dim,action_dim,max_action)

    save_path_eval = os.path.join(agent_name,'eval/')
    os.makedirs(save_path_eval, exist_ok=True)
    agent.load(policy_file)
    evaluate_agent(agent,env,config['eval_episodes'],config_eval,save_path_eval)
        

if __name__=='__main__':
    p_target = [2,0]
    p_0 = np.zeros(2)
    p_1 = [1/2,1]
    nb_points_path = 500
    path = generate_demi_circle_path(p_0,p_target,nb_points_path)
    config = {
        'x_0' : np.zeros(2),
        'C' : 1,
        'D' : 0.1,
        'threshold' : 0.02,
        't_max': 5,
        'n_t_sim':200,
        't_init':0,
        'steps_per_action':5,
        'nb_episode':700,
        'batch_size':256,
        'eval_freq':100,
        'save_model':True,
        'eval_episodes' : 3,
        'path' : path,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :100,
        'load_model':"",
        'episode_per_update' :3,
        'discount_factor' : 1,
        'beta':0.18
    }
    #run_expe(config)
    ### EVAL ## 
    evaluate_after_training('agent_TD3_2025-04-07_16-44/')


