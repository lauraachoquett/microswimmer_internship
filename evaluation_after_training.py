import numpy as np 
from generate_path import *
from math import cos,sin
import os
import TD3
from evaluate_agent import evaluate_agent
from env_swimmer import MicroSwimmer
import pickle
from scipy.spatial import KDTree
from invariant_state import coordinate_in_global_ref
colors = plt.cm.tab10.colors
import matplotlib
from distance_to_path import min_dist_closest_point
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
    plot_robust_D(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)
    config_eval['D']=D
    plot_robust_u_bg_uniform(config_eval,file_name_or,agent,env,save_path_eval,15,threshold)

def plot_robust_D(config_eval,file_name_or,agent,env,save_path_eval,nb_D,threshold):
    save_path_eval_D = os.path.join(save_path_eval,'robust_D_bis/')
    if not os.path.exists(save_path_eval_D):
            os.makedirs(save_path_eval_D)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    D_values = np.linspace(0.0, config_eval['Dt_action'], nb_D) 
    config_eval_bis=config
    for idx_tr,thr in enumerate(threshold):
        mean_reward_D = np.zeros(nb_D)
        std_reward_D = np.zeros(nb_D)
        success_rate_D = np.zeros(nb_D)
        config_eval_bis['threshold']=thr
        for idx, D in enumerate(D_values):
            print(f"iter : {idx}")
            config_eval_bis['D'] = D
            file_name = file_name_or + f'_{idx}'
            mean_reward, std_reward, success_rate = evaluate_agent(agent, env, config_eval_bis['eval_episodes'],
                                                                    config_eval_bis, save_path_eval_D,
                                                                    file_name=file_name, random_parameters=False)
            mean_reward_D[idx] = mean_reward
            std_reward_D[idx] = std_reward
            success_rate_D[idx] = success_rate

        plot_mean_reward_success_rate(mean_reward_D,std_reward_D,success_rate_D,D_values * 2/config_eval['Dt_action'],r'$\frac{2D}{U^2 \Delta t}$',idx_tr,thr,fig,axs)

    path_save_fig_D = os.path.join(save_path_eval_D,f'success_rate_rew_D')
    fig.suptitle(f"episodes : {config_eval['eval_episodes']}", fontsize=10)
    fig.tight_layout()
    plt.savefig(path_save_fig_D,dpi=400,bbox_inches='tight')
    

def plot_robust_u_bg_uniform(config_eval,file_name_or,agent,env,save_path_eval,nb_norm,threshold):
    dir_d = {
        'North': np.array([0,1]),
        'West':np.array([-1,0]),
        'South' :np.array([0,-1]),
        'East':np.array([1,0]),
        }
    norm_values = np.linspace(0.1,0.7,nb_norm)
    config_eval_dir = config_eval
    for dir,vec in dir_d.items():
        save_path_eval_dir = os.path.join(save_path_eval,dir)
        print(dir)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for idx_tr,thr in enumerate(threshold):
            config_eval_dir['threshold'] = thr
            mean_reward_D = np.zeros(nb_norm)
            std_reward_D = np.zeros(nb_norm)
            success_rate_D = np.zeros(nb_norm)
            if not os.path.exists(save_path_eval_dir):
                os.makedirs(save_path_eval_dir)
            for idx,norm in enumerate(norm_values):
                config_eval_dir['u_bg'] = vec * norm 
                file_name = file_name_or + f'_{idx}'
                mean_reward, std_reward, success_rate = evaluate_agent(agent, env, config_eval_dir['eval_episodes'],
                                                                    config_eval_dir, save_path_eval_dir,
                                                                    file_name=file_name, random_parameters=False)
                mean_reward_D[idx] = mean_reward
                std_reward_D[idx] = std_reward
                success_rate_D[idx] = success_rate

            plot_mean_reward_success_rate(mean_reward_D,std_reward_D,success_rate_D,norm_values,r'$ \frac{\|u\|}{U}$',idx_tr,thr,fig,axs)
        path_save_fig_D = os.path.join(save_path_eval_dir,f'success_rate_rew_u_bg')
        fig.suptitle(f"episodes : {config_eval_dir['eval_episodes']}", fontsize=10)
        fig.tight_layout()
        plt.savefig(path_save_fig_D,dpi=400,bbox_inches='tight')



def plot_mean_reward_success_rate(mean_rewards, std_rewards, list_success_rate, abscisse, xlabel, idx_tr, thr, fig, axs):
    axs[0].plot(abscisse, mean_rewards, color=colors[idx_tr], label=r"$\delta {:.3f}$".format(thr))
    axs[0].fill_between(abscisse,
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        alpha=0.3, color=colors[idx_tr])
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    
    axs[1].plot(abscisse, list_success_rate, color=colors[idx_tr], label=r"$\delta {:.3f}$".format(thr))
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel('Success rate')
    axs[1].legend()

def policy_direction(agent_name,config):
    p_0 = np.zeros(2)
    p_target = np.array([1/8,1/64])
    k=-0.4
    nb_points=100
    path = generate_curve(p_0,p_target,k,nb_points)
    tree = KDTree(path)
    x_0 = np.array([0.02,-0.01])
    x = x_0
    config_eval['x_0'] = x_0
    env = MicroSwimmer(config_eval['x_0'],config_eval['C'],config['Dt_action']/config['steps_per_action'])
    beta =config['beta']
    state,done = env.reset(tree,path),False
    u_bg = np.zeros(2)
    D = config['D']
    threshold = config['threshold']

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



def plot_action(path,x,p_0,id_cp,action,id):
    plt.scatter(x[0],x[1],color = colors[id%10])
    plt.annotate(f'{id}',  # texte en LaTeX
             xy=(x[0], x[1]),         
             xytext=(x[0], x[1]+1/(64*20)))
    plt.scatter(path[id_cp,0],path[id_cp,1],color = colors[id%10],marker='*')
    plt.quiver(x[0],x[1],action[0],action[1],scale = 20,width=0.005, color='grey')
    plt.xlabel('x')
    plt.ylabel('y')

    
if __name__=='__main__':
    agent = 'agents/agent_TD3_2025-04-10_13-59'
    path_config = os.path.join(agent,'config.pkl')
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
    #policy_direction(agent,config_eval)
    evaluate_after_training(agent,config_eval,file_name_or=f"eval_with_ondulating_path",type='ondulating')

    plt.close()
    agent = 'agents/agent_TD3_2025-04-10_11-26'
    path_config = os.path.join(agent,'config.pkl')
    print(os.path.getsize(path_config))
    with open(path_config, "rb") as f:
        config = pickle.load(f)
    print("training with rankine bg : ", config['rankine_bg'])
    print("training with uniform bg : ", config['uniform_bg'])
    print("training with random curve :", config['random_curve'] if 'random_curve' in config else "False") 

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
    evaluate_after_training(agent,config_eval,file_name_or=f"eval_with_ondulating_path",type='ondulating')