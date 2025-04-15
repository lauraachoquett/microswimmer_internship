import numpy as np 
from generate_path import *
from math import cos,sin
import os
import TD3
from env_swimmer import MicroSwimmer
import pickle
import json
from scipy.spatial import KDTree
colors = plt.cm.tab10.colors
from evaluate_agent import evaluate_agent
from visualize import visualize_streamline,plot_robust_D,plot_robust_u_bg_uniform,plot_robust_u_bg_rankine
from distance_to_path import min_dist_closest_point
from invariant_state import coordinate_in_global_ref
from plot import plot_action
from statistics import mean
from rank_agents import rank_agents_by_rewards
import random
from pathlib import Path

def format_sci(x):
    return "{:.3e}".format(x)

def evaluate_after_training(agent_files,file_name_or,p_target,p_0,seed=42,type=None,translation=None,theta=None,dir=None,norm=None,a=None,center=None,cir=None,title_add=''):
    np.random.seed(seed)
    random.seed(seed)
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
        
    p_target = R @ p_target + translation
    p_0 = R @ p_0 + translation
    p_1 = [1/4,-1/4] + translation
    nb_points_path = 500
    k=0
    if type=='line':
        path,_ = generate_simple_line(p_0,p_target,nb_points_path)
    if type =='two_line':
        path,_ = generate_line_two_part(p_0,p_1,p_target,nb_points_path)
    if type =='circle' : 
        path,_ = generate_demi_circle_path(p_0,p_target,nb_points_path)
    if type=='ondulating':
        path = generate_random_ondulating_path(p_0,p_target,nb_points_path,amplitude = 0.5,frequency=2)
    if type =='curve':
        k=-4
        path = generate_curve(p_0,p_target,k,nb_points_path)
    tree=  KDTree(path)
    
    results={}
    
    uniform_bg=False
    rankine_bg=False
    if dir is not None and norm is not None:
        dir = np.array(dir)
        norm=norm
        parameters  = [dir,norm]
        uniform_bg = True
    elif a is not None and center is not None and cir is not None:
        parameters = [center,a,cir]
        rankine_bg = True
    else : 
        parameters = []
        
    file_path_result = "results_evaluation/"
    os.makedirs(file_path_result, exist_ok=True)
    file_name = os.path.join(file_path_result,file_name_or)

    try :
        with open(file_name,"r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results={}
    results['type'] = [type]
    
    for agent_name in agent_files:

        print("Agent name : ", agent_name)
        config_eval = initialize_parameters(agent_name,p_target,p_0)
        training_type = {'rankine_bg':config_eval['rankine_bg'],'uniform_bg':config_eval['uniform_bg'],'random_curve':config_eval['random_curve'],'velocity_bool':config_eval['velocity_bool'],'load_model':config_eval['load_model'],'n_lookahead':config_eval['n_lookahead']}
        if agent_name in results.keys():
            results[agent_name]['training type'] = training_type
            print(f"Agent {agent_name} already evaluated.")
            continue
        config_eval['uniform_bg'] = uniform_bg
        config_eval['rankine_bg'] = rankine_bg
        config_eval['random_curve']=False

        Dt_action = config_eval['Dt_action']
        steps_per_action = config_eval['steps_per_action']
        Dt_sim= Dt_action/steps_per_action
        threshold = config_eval['threshold']

        # print("Curvature max du chemin : ", format_sci(np.max(courbures(path))))
        config_eval['path']=path
        config_eval['tree'] = tree
        env = MicroSwimmer(config_eval['x_0'],config_eval['C'],Dt_sim,config_eval['velocity_bool'],config_eval['n_lookahead'],seed)

        state_dim=env.observation_space.shape[0]
        action_dim=env.action_space.shape[0]
        max_action= float(env.action_space.high[0])
        agent = TD3.TD3(state_dim,action_dim,max_action)

        save_path_eval = os.path.join(agent_name,'eval_bg/')
        os.makedirs(save_path_eval, exist_ok=True)
        
        policy_file = os.path.join(agent_name,'models/agent')
        agent.load(policy_file)
        rewards_per_episode, rewards_t_per_episode, rewards_d_per_episode,success_rate,_=evaluate_agent(agent,env,config_eval['eval_episodes'],config_eval,save_path_eval,f'eval_with_{title_add}_{type}',False,title='',plot=True,parameters=parameters,plot_background=True)
        
        
        results[agent_name] = {
            'rewards': rewards_per_episode,
            'rewards_time': rewards_t_per_episode,
            'rewards_distance': rewards_d_per_episode,
            'success_rate': success_rate,
            'n_eval_episodes': config_eval['eval_episodes'],
            'training type':training_type
        }
        print('-----------------------------------------------')
        print("Success rate : ",success_rate)
        print("Mean rewards : ",format_sci(mean(rewards_per_episode)))
        print("Mean rewards t : ",format_sci(mean(rewards_t_per_episode)))
        print("Mean rewards d : ",format_sci(mean(rewards_d_per_episode)))
        print('-----------------------------------------------')
        visualize_streamline(agent,config_eval,f'streamline_{title_add}_{type}',save_path_eval,type=type,title='',k=k,parameters=parameters)

        
    

    file_name = os.path.join(file_path_result,file_name_or)
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)    
        

    threshold=[0.07]
    D =config_eval['D']
    #plot_robust_D(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    config_eval['D']=D
    #plot_robust_u_bg_uniform(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    #plot_robust_u_bg_rankine(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    plt.close()
    return results

def policy_direction(agent_name,config_eval):
    p_0 = np.zeros(2)
    p_target = np.array([1/8,1/64])
    k=-0.4
    nb_points=100
    path = generate_curve(p_0,p_target,k,nb_points)
    tree = KDTree(path)
    x_0 = np.array([0.02,-0.01])
    x = x_0
    config_eval['x_0'] = x_0
    env = MicroSwimmer(config_eval['x_0'],config_eval['C'],config_eval['Dt_action']/config_eval['steps_per_action'])
    beta =config_eval['beta']
    state,done = env.reset(tree,path),False
    u_bg = np.zeros(2)
    D = config_eval['D']
    threshold = config_eval['threshold']

    policy_file = os.path.join(agent_name,'models/agent')
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    agent = TD3.TD3(state_dim,action_dim,max_action)
    agent.load(policy_file)

    nb_steps = 5
    state,done = env.reset(tree,path),False

    plt.plot(path[:,0],path[:,1],label='path', color='black', linewidth=2)
    plt.scatter(p_0[0],p_0[1],color='black')
    for i in range(nb_steps):
        action = agent.select_action(state)
        d,id_cp = min_dist_closest_point(x,tree)
        dir_path = (path[id_cp+1]-path[id_cp])
        action_global = coordinate_in_global_ref(path[id_cp],dir_path,action)
        plot_action(path,x,p_0,id_cp,action_global,i)
        next_state,x,reward,done,_ = env.step(action,tree,path,p_target,beta,D,u_bg,threshold)
        state = next_state
    save_path_eval_action = os.path.join(agent_name,'eval_bg/action_choice')

    plt.legend()
    plt.savefig(save_path_eval_action,dpi=200,bbox_inches='tight')

def initialize_parameters(agent_file,p_target,p_0):
    path_config = os.path.join(agent_file,'config.pkl')
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    # print("Training with rankine bg : ", config['rankine_bg'])
    # print("Training with uniform bg : ", config['uniform_bg'])
    # print("Training with random curve :", config['random_curve'] if 'random_curve' in config else "False") 
    # print("Training with velocity in its state :", config['velocity_bool'] if 'velocity_bool' in config else "False") 
    # print("Retrained on : ",config['load_model'])
    config_eval =config
    config_eval['random_curve'] =  config['random_curve'] if 'random_curve' in config else False
    config_eval['p_target'] = p_target
    config_eval['p_0'] = p_0
    config_eval['x_0'] = p_0
    config_eval['nb_points_path']=500
    config_eval['t_max'] =12
    config_eval['eval_episodes']=100
    config_eval['velocity_bool'] =  config['velocity_bool'] if 'velocity_bool' in config else False
    config_eval['beta'] = 0.25
    config_eval['Dt_action'] = config_eval['Dt_action'] if 'Dt_action' in config else 1/30
    config_eval['n_lookahead'] = config['n_lookahead'] if 'n_lookahead' in config else 5
    maximum_curvature = 30
    l = 1/maximum_curvature
    Dt_action = 1/maximum_curvature
    threshold = 0.07
    D = threshold**2/(20*Dt_action)
    config_eval['D'] = D
    return config_eval
    

if __name__=='__main__':
    agent_file_1 = 'agents/semi-circle_u_bg/agent_TD3_2025-04-08_15-07_eval'
    agent_file_3 = 'agents/retrained/agent_TD3_2025-04-10_11-26'
    agent_file_4 = 'agents/state_velocity/agent_TD3_2025-04-10_13-59'
    agent_file_5 = 'agents/state_velocity/agent_TD3_2025-04-11_13-37'
    agent_file_6 = 'agents/state_velocity/agent_TD3_2025-04-11_14-01_retrained'
    agent_file_7 = 'agents/state_velocity/agent_TD3_2025-04-11_14-28'

    
    agents_file = [agent_file_1,agent_file_3,agent_file_4,agent_file_5,agent_file_6,agent_file_7]
    directory_path = Path("agents/")
    
    for item in directory_path.iterdir():
        if item.is_dir() and "agent_TD3" in item.name :
            agents_file.append(os.path.join(directory_path, item.name))
            
    print("Agents files : ",agents_file)
    types = ['circle','curve','ondulating','line','two_line']
    title_add = 'rankine_a_05__cir_3_center_1_075'
    print("---------------------Evaluation with rankine bg---------------------")
    for type in types[:-1]:
        a= 0.5
        cir = 2
        center = np.array([1,3/4])
        results = evaluate_after_training(agents_file,f'result_evaluation_{title_add}_{type}.json',type=type,p_target = [2,0],p_0 = [0,0],title_add=title_add,a=a,center=center,cir=cir)

    norm=0.5
    dict = {
        'east_05': np.array([1,0]),
        'west_05': np.array([-1,0]),
        'north_05': np.array([0,1]),
        'south_05': np.array([0,-1]),
    }
    print("---------------------Evaluation with uniform bg---------------------")
    for type in types[:-1]:
        for title_add,dir in dict.items():
            results = evaluate_after_training(agents_file,f'result_evaluation_{title_add}_{type}.json',type=type,p_target = [2,0],p_0 = [0,0],title_add=title_add,dir=dir,norm=norm)
            rank_agents_by_rewards(results)

    
    print("---------------------Evaluation with no bg---------------------")
    title_add = 'free'
    for type in types[:-1]:
        results = evaluate_after_training(agents_file,f'result_evaluation_{title_add}_{type}.json',type=type,p_target = [2,0],p_0 = [0,0],title_add=title_add)
        rank_agents_by_rewards(results)
