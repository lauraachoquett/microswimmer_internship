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
from analyze_results import rank_agents_by_rewards
import random

def format_sci(x):
    return "{:.3e}".format(x)

def evaluate_after_training(agent_files,file_name_or,p_target,p_0,seed=42,type=None,translation=None,theta=None):
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
    
    if type =='two_line':
        path = generate_line_two_part(p_0,p_1,p_target,nb_points_path)
    if type =='circle' : 
        path = generate_demi_circle_path(p_0,p_target,nb_points_path)
    if type=='ondulating':
        path = generate_random_ondulating_path(p_0,p_target,nb_points_path,amplitude = 0.5,frequency=2)
    tree=  KDTree(path)
    
    results={}
    dir = []
    norm = 0
    parameters  = [dir,norm]
    results['type'] = ['ondulating','free',parameters]
    for agent_name in agent_files:
        print('##############################################')
        print("Agent name : ", agent_name)
        config_eval = initialize_parameters(agent_name)
        config_eval['uniform_bg'] = False

        Dt_action = config_eval['Dt_action']
        steps_per_action = config_eval['steps_per_action']
        Dt_sim= Dt_action/steps_per_action
        threshold = config_eval['threshold']

        print("Curvature max du chemin : ", format_sci(np.max(courbures(path))))
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
        rewards_per_episode, rewards_t_per_episode, rewards_d_per_episode,success_rate,_=evaluate_agent(agent,env,config_eval['eval_episodes'],config_eval,save_path_eval,'eval_with_ondulating',False,title='',plot=True,parameters=[],plot_background=False)
        
        
        training_type = {'rankine_bg':config_eval['rankine_bg'],'uniform_bg':config_eval['uniform_bg'],'random_curve':config_eval['random_curve'],'velocity_bool':config_eval['velocity_bool']}
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
        visualize_streamline(agent,config_eval,'streamline_ondulating_path',save_path_eval,u_bg=np.array([0.0,0]),type='ondulating',title='')

        
    
    file_path_result = "results_evaluation/"
    os.makedirs(file_path_result, exist_ok=True)
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

def initialize_parameters(agent_file):
    path_config = os.path.join(agent_file,'config.pkl')
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    print("Training with rankine bg : ", config['rankine_bg'])
    print("Training with uniform bg : ", config['uniform_bg'])
    print("Training with random curve :", config['random_curve'] if 'random_curve' in config else "False") 
    print("Training with velocity in its state :", config['velocity_bool'] if 'velocity_bool' in config else "False") 
    print("Retrained on : ",config['load_model'])
    config_eval =config
    config_eval['p_target'] = [2,2]
    config_eval['random_curve']=False
    config_eval['nb_points_path']=200
    config_eval['t_max'] =20
    config_eval['eval_episodes']=10
    config_eval['rankine_bg'] = False
    config_eval['uniform_bg'] = False
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
    agent_file_2 = 'agents/agent_TD3_2025-04-11_16-19'
    agent_file_3 = 'agents/retrained/agent_TD3_2025-04-10_11-26'
    agent_file_4 = 'agents/state_velocity/agent_TD3_2025-04-10_13-59'
    agent_file_5 = 'agents/state_velocity/agent_TD3_2025-04-11_13-37'
    agent_file_6 = 'agents/state_velocity/agent_TD3_2025-04-11_14-01_retrained'
    agent_file_7 = 'agents/state_velocity/agent_TD3_2025-04-11_14-28'
    agent_file_8 = 'agents/agent_TD3_2025-04-11_16-09'
    agent_file_9 = 'agents/agent_TD3_2025-04-11_16-37'
    agent_file_10 = 'agents/agent_TD3_2025-04-14_15-15'
    agent_file_11 = 'agents/agent_TD3_2025-04-14_15-33'
    agent_file_12 = 'agents/agent_TD3_2025-04-14_15-42'
    agent_file_13 = 'agents/agent_TD3_2025-04-14_16-45'
    agents_file = [agent_file_13,agent_file_10,agent_file_12,agent_file_11,agent_file_1,agent_file_2,agent_file_3,agent_file_4,agent_file_5,agent_file_6,agent_file_7,agent_file_8,agent_file_9]
    

    results = evaluate_after_training(agents_file,'result_evaluation_usual_setup.json',type='ondulating',p_target = np.array([2,2]),p_0 = np.array([0,0]))
    rank_agents_by_rewards(results)
    