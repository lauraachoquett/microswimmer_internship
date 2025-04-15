import numpy as np
import torch
import argparse
import os
from utils import ReplayBuffer,random_bg_parameters
import TD3
from datetime import datetime 
from env_swimmer import MicroSwimmer
from generate_path import *
import matplotlib.pyplot as plt
from invariant_state import *
from math import sqrt
import pickle
from sde import rankine_vortex,uniform_velocity
from evaluate_agent import evaluate_agent
from scipy.spatial import KDTree
colors = plt.cm.tab10.colors
from statistics import mean

def format_sci(x):
    return "{:.3e}".format(x)


def run_expe(config,agent_file='agents'):
    print(" --------------- TRAINING ---------------")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = os.path.join(agent_file, f"agent_TD3_{timestamp}")

    os.makedirs(file_name, exist_ok=True)
    with open(os.path.join(file_name, 'config.pkl'), "wb") as f:
        pickle.dump(config, f)


    path = config['path']
    p_target = config['p_target']
    x_0 = config['x_0']
    p_0 = config['p_0']
    tree = config['tree']
    print(f'Starting point : {p_0}')
    print(f'Target point : {path[-1]}')

    C = config['C']
    t_max =config['t_max']
    n_lookahead = config['n_lookahead']
    steps_per_action=config['steps_per_action']
    Dt_action =  config['Dt_action']
    Dt_sim = Dt_action /steps_per_action

    threshold = config['threshold']
    Dt_action = Dt_sim*steps_per_action
    beta = config['beta']


    env = MicroSwimmer(x_0,C,Dt_sim,config['velocity_bool'],n_lookahead)

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

    replay_buffer = ReplayBuffer(state_dim,action_dim)

    save_path_result = f"./{file_name}/results"
    save_path_result_fig = os.path.join(save_path_result,'fig/')
    if not os.path.exists(save_path_result):
        os.makedirs(save_path_result)
        os.makedirs(save_path_result_fig)
    save_path_model = f"./{file_name}/models/agent"
    if save_model and not os.path.exists(save_path_model):
        os.makedirs(save_path_model)

    agent_to_load = config['load_model']

    if agent_to_load != '':
        policy_file = os.path.join(config['load_model'],'models/agent')
        agent.load(policy_file)
        path_config = os.path.join(config['load_model'],'config.pkl')
        with open(path_config, "rb") as f:
            config = pickle.load(f)
        print("Policy loaded !")
        episode_start_update= eval_freq*2
        episode_update = 2
        config['random_curve'] = True
    


    iter = 0
    episode_reward = 0
    episode_num = 0
    training_reward=[]
    best_eval_result=-np.inf

    #random_eval = evaluate_agent(agent,env,eval_episodes,config,save_path_result)
    #print(f"Initial random evaluation : {random_eval}")
    state,done = env.reset(tree,path),False
    count_reach_target=0

    x=p_0
    u_bg = np.zeros(2)
    D = config['D']

    if config['random_curve']:
        k = 2
        path = generate_curve(p_0,p_target,k,config['nb_points_path'])
        tree = KDTree(path)
        

    ########### TRAINING LOOP ###########
    while episode_num <nb_episode:
        iter+=1

        if iter%steps_per_action==0 or iter==1:
            action = agent.select_action(state)

        if episode_num>config['pertubation_after_episode']: 
            if config['uniform_bg']:
                u_bg = uniform_velocity(dir,norm)
            if config['rankine_bg']:
                u_bg = rankine_vortex(x,a,center,cir)

        next_state,reward,done,info = env.step(action,tree,path,p_target,beta,D,u_bg,threshold)
        x= info['x']
        replay_buffer.add(state.flatten(), action, next_state.flatten(), reward, done)
            

        state = next_state
        episode_reward += reward

        if  episode_num%episode_update==0 and episode_num>episode_start_update:
            agent.train(replay_buffer, batch_size)
        if done:
            count_reach_target+=1
            # if agent_to_load != '':
                #beta = beta * 1.001
        if done or iter*Dt_sim> t_max: 
            state,done =env.reset(tree,path),False
            training_reward.append(episode_reward)

            if episode_num%eval_freq==0 and episode_num > 10:
                print(f"Total iter: {iter+1} Episode Num: {episode_num+1} Reward: {episode_reward:.3f} Success rate: {count_reach_target/eval_freq}")
                path_save_fig= os.path.join(save_path_result_fig,"training_reward.png")
                eval_rew,_,_,_,_= evaluate_agent(agent,env,eval_episodes,config,save_path_result_fig,"eval_during_training",False)
                #evaluations.append(eval)
                eval_rew = mean(eval_rew)
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


                if agent_to_load:
                    print("Beta increased : ",beta)
                    
                count_reach_target = 0 


            iter = 0
            episode_reward=0
            episode_num+=1
            dir, norm,center,a ,cir=random_bg_parameters()
            if episode_num>config['pertubation_after_episode']:
                if config['random_curve']:
                    k = (np.random.rand()-0.5)*4
                    path = generate_curve(p_0,p_target,k,config['nb_points_path'])
                    tree = KDTree(path)


    agent.save(os.path.join(save_path_model,'last'))



if __name__=='__main__':
    p_target = [2,0]
    p_0 = np.zeros(2)
    nb_points_path = 500
    path,d = generate_demi_circle_path(p_0,p_target,nb_points_path)
    print("Distance path points:      ",format_sci(np.linalg.norm(path[1,:]-path[0,:])))
    tree = KDTree(path)
    print("Curvature max du chemin :  ", format_sci(np.max(courbures(path))))
    t_max = 8
    t_init= 0
    maximum_curvature = 30
    l = 1/maximum_curvature
    Dt_action = 1/maximum_curvature
    threshold = 0.07
    D = threshold**2/(20*Dt_action)
    print("D:                         ", format_sci(D))
    print("Dt_action:                 ", format_sci(Dt_action))
    print("Threshold:                 ", format_sci(threshold))
    print("Mean diffusion distance:   ", format_sci(sqrt(2 * Dt_action * D)))
    print("Distance during Dt_action: ", format_sci(Dt_action))
    print("Distance to cover:         ", format_sci(d))
    print("Expected precision:        ", format_sci(threshold / d))
    config = {
        'x_0' : p_0,
        'C' : 1,
        'D' : D,
        'u_bg' : np.array([0,1])*0.0,
        'threshold' :threshold,
        't_max': t_max,
        't_init':t_init,
        'steps_per_action':5,
        'nb_episode':500,
        'batch_size':128,
        'eval_freq':50,
        'save_model':True,
        'eval_episodes' : 3,
        'episode_start_update' : 10,
        'path' : path,
        'tree' :tree,
        'p_0' : p_0,
        'p_target' : p_target,
        'start_timesteps' :100,
        'load_model':"",
        'episode_per_update' :5,
        'discount_factor' : 1,
        'beta':0.25,
        'uniform_bg':False,
        'rankine_bg':True,
        'pertubation_after_episode' :150,
        'random_curve' : False,
        'nb_points_path':500,
        'Dt_action': Dt_action,
        'velocity_bool' : True,
        'n_lookahead' : 5,
    }
    run_expe(config)