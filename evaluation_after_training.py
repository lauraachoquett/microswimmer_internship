import numpy as np 
from generate_path import *
from math import cos,sin
import os
import TD3
from env_swimmer import MicroSwimmer
import pickle
from scipy.spatial import KDTree
colors = plt.cm.tab10.colors

from visualize import visualize_streamline,plot_robust_D,plot_robust_u_bg_uniform,plot_robust_u_bg_rankine
from distance_to_path import min_dist_closest_point
from invariant_state import coordinate_in_global_ref
from plot import plot_action

def format_sci(x):
    return "{:.3e}".format(x)

def evaluate_after_training(agent_name,config_eval,file_name_or,type=None,translation=None,theta=None):
    Dt_action = config_eval['Dt_action']
    steps_per_action = config_eval['steps_per_action']
    Dt_sim= Dt_action/steps_per_action
    threshold = config['threshold']
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
        
    p_target = config_eval['p_target']
    p_0 = config_eval['p_0']
    p_target = R@p_target+translation
    p_0 = R@p_0+translation
    p_1 = [1/4,-1/4]+translation
    nb_points_path = config_eval['nb_points_path']

    path = config_eval['path']

    if type =='two_line':
        path = generate_line_two_part(p_0,p_1,p_target,nb_points_path)
    if type =='circle' : 
        path = generate_demi_circle_path(p_0,p_target,nb_points_path)
    if type=='ondulating':
        path = generate_random_ondulating_path(p_0,p_target,nb_points_path,amplitude = 0.5,frequency=2)


    print("Curvature max du chemin : ", format_sci(np.max(courbures(path))))
    config_eval['path']=path
    config_eval['tree'] = KDTree(path)
    policy_file = os.path.join(agent_name,'models/agent')

    env = MicroSwimmer(config_eval['x_0'],config_eval['C'],Dt_sim,config_eval['velocity_bool'])

    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    max_action= float(env.action_space.high[0])
    agent = TD3.TD3(state_dim,action_dim,max_action)

    save_path_eval = os.path.join(agent_name,'eval_bg/')
    os.makedirs(save_path_eval, exist_ok=True)
    agent.load(policy_file)

    threshold=[0.07]
    D =config_eval['D']
    #plot_robust_D(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    config_eval['D']=D
    #plot_robust_u_bg_uniform(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    visualize_streamline(agent,config_eval,file_name_or,save_path_eval,type='ondulating')
    #plot_robust_u_bg_rankine(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)

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


if __name__=='__main__':
    agent_file = 'agents/agent_TD3_2025-04-11_16-19'
    path_config = os.path.join(agent_file,'config.pkl')
    print(os.path.getsize(path_config))
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    print("training with rankine bg : ", config['rankine_bg'])
    print("training with uniform bg : ", config['uniform_bg'])
    print("training with random curve :", config['random_curve'] if 'random_curve' in config else "False") 

    config_eval =config
    config_eval['p_target'] = [2,2]
    config_eval['random_curve']=False
    config_eval['nb_points_path']=200
    config_eval['t_max'] =12
    config_eval['eval_episodes']=3
    config_eval['rankine_bg'] = False
    config_eval['uniform_bg'] = False
    config_eval['velocity_bool'] = True
    
    #policy_direction(agent,config_eval)
    plt.close()
    evaluate_after_training(agent_file,config_eval,file_name_or=f"eval_with_ondulating_path",type='ondulating')

    plt.close()
    agent_file = 'agents/retrained/agent_TD3_2025-04-10_11-26'
    path_config = os.path.join(agent_file,'config.pkl')
    print(os.path.getsize(path_config))
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    print("training with rankine bg : ", config['rankine_bg'])
    print("training with uniform bg : ", config['uniform_bg'])
    print("training with random curve :", config['random_curve'] if 'random_curve' in config else "False") 
    print("retrained : ", config['load_model'])
    config_eval =config
    config_eval['velocity_bool'] = False
    config_eval['p_target'] = [2,2]
    config_eval['random_curve']=False
    config_eval['nb_points_path']=200
    config_eval['t_max'] =12
    config_eval['eval_episodes']=3
    config_eval['rankine_bg'] = False
    config_eval['uniform_bg'] = False

    #policy_direction(agent,config_eval)
    #evaluate_after_training(agent_file,config_eval,file_name_or=f"eval_with_ondulating_path",type='ondulating')