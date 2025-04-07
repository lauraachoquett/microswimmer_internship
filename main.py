import numpy as np
import torch
import argparse
import os
from statistics import mean
import utils
import TD3
from datetime import datetime 
from env_swimmer import MicroSwimmer
from generate_path import generate_simple_line
import matplotlib.pyplot as plt
from invariant_state import *

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
    state,done = env.reset(p_0,T_0),False
    states_episode = []
    states_list_per_episode=[]
    while episode_num <eval_episodes:
        states_episode.append(coordinate_in_global_ref(p_0,T_0,state))
        iter+=1
        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        next_state,reward,done,_ = env.step(action,path,p_target,p_0,T_0)
        state = next_state
        episode_reward += reward
        rewards.append(episode_reward)
        
        if done or iter*Dt_sim> t_max: 
            states_list_per_episode.append(np.array(states_episode))
            state,done =env.reset(p_0,T_0),False
            iter = 0
            episode_reward=0
            episode_num+=1
            states_episode=[]

    plt.close()
    path_save_fig= os.path.join(save_path_result_fig,"training_eval_trajectories.png")
    for list_state in states_list_per_episode :
        indices = np.linspace(0, len(path) - 1, n_t_sim).astype(int)
        path = path[indices]
        #path_local_ref = np.zeros_like(path)
        #for i in range(len(path)):
            #path_local_ref[i] = (coordinate_in_path_ref(p_0,T_0,path[i]))
        plt.plot(path[:,0],path[:,1],label='path')
        plt.plot(list_state[:,0],list_state[:,1])
        plt.scatter(list_state[-1,0],list_state[-1,1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Trajectories ")
        plt.savefig(path_save_fig,dpi=100,bbox_inches='tight')
            

    return mean(rewards)

def run_expe(config):
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"agent_TD3_{timestamp}"


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
    print(f"Delta de temps entre chaque action : {Dt_action}")

    env = MicroSwimmer(x_0,C,Dt_sim,threshold)

    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    nb_episode = config['nb_episode']
    batch_size = config['batch_size']
    eval_freq = config['eval_freq']
    eval_episodes = config['eval_episodes']
    save_model = config['save_model']
    
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
    state,done = env.reset(p_0,T_0),False
    print("Beginning of the training...")
    print(f"state : {state}")
    count_reach_target=0
    while episode_num <nb_episode:
        iter+=1

        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        next_state,reward,done,_ = env.step(action,path,p_target,p_0,T_0)
        replay_buffer.add(state, action, next_state, reward, done)
            

        state = next_state
        episode_reward += reward

        if iter >=  config['start_timesteps']:
            agent.train(replay_buffer, batch_size)
        if done:
            count_reach_target+=1
        if done or iter*Dt_sim> t_max: 
            state,done =env.reset(p_0,T_0),False
            training_reward.append(episode_reward)

            if episode_num%eval_freq==0:
                print(f"Total iter: {iter+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f} Counter target reached: {count_reach_target}")
                path_save_fig= os.path.join(save_path_result_fig,"training_reward.png")
                eval = evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig)
                evaluations.append(eval)
                print(f"Eval result : {eval}")
                np.save(save_path_result, evaluations)
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

            



if __name__=='__main__':
    p_target = np.ones(2)
    p_0 = np.zeros(2)
    nb_points_path = 1000
    path = generate_simple_line(p_0,p_target,nb_points_path)
    config = {
        'x_0' : np.zeros(2),
        'C' : 1,
        'D' : 0.1,
        'threshold' : 0.05,
        't_max': 5,
        'n_t_sim':200,
        't_init':0,
        'steps_per_action':5,
        'nb_episode':500,
        'batch_size':256,
        'eval_freq':100,
        'save_model':True,
        'eval_episodes' : 5,
        'path' : path,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :10,
        'load_model':""
    }
    #run_expe(config)
    ### EVAL ###

    translation = np.ones(2)*1

    theta = np.pi/8 
    sin_th = sin(theta)
    cos_th = cos(theta)
    R = np.array([[cos_th, -sin_th],
                  [sin_th,  cos_th]])
    
    p_target = np.ones(2) + translation
    p_0 = np.zeros(2) + translation
    T_0=p_target
    nb_points_path = 1000

    path = generate_simple_line(p_0,p_target,nb_points_path)
    path_local_ref = np.zeros_like(path)
    for i in range(len(path)):
        path_local_ref[i] = (coordinate_in_path_ref(p_0,T_0,path[i]))
    plt.plot(path_local_ref[:,0],path_local_ref[:,1],label='path_local_ref')
    plt.plot(path[:,0],path[:,1],label='path')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Path : line + transformation")
    plt.legend()
    plt.savefig(f"fig/path_test_transformatino.png",dpi=100,bbox_inches='tight')

    config_eval = {
        'x_0' : p_0,
        'C' : 1,
        'D' : 0.1,
        'threshold' : 0.05,
        't_max': 5,
        'n_t_sim':200,
        't_init':0,
        'steps_per_action':5,
        'nb_episode':500,
        'batch_size':256,
        'eval_freq':100,
        'save_model':True,
        'eval_episodes' : 5,
        'path' : path,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :10,
        'load_model':""
    }
    
    policy_file = 'agent_TD3_2025-04-07_09-19/models'
    n_t_sim = config_eval['n_t_sim']
    t_init =config_eval['t_init']
    t_max =config_eval['t_max']
    Dt_sim = (t_max-t_init)/n_t_sim
    env = MicroSwimmer(config_eval['x_0'],config_eval['C'],Dt_sim,config_eval['threshold'])

    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    agent = TD3.TD3(state_dim,action_dim,max_action)
    save_path_eval =  'agent_TD3_2025-04-07_09-19/eval'
    os.makedirs(save_path_eval, exist_ok=True)
    agent.load(policy_file)
    evaluate_agent(agent,env,5,config_eval,save_path_eval)






