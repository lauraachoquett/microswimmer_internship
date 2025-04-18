import gymnasium as gym
import numpy as np
from invariant_state import *
from distance_to_path import min_dist_closest_point
from sde import solver

class MicroSwimmer(gym.Env):
    def __init__(self,x_0,C,Dt,velocity_bool,n_lookahead=5,seed =None):
        super(MicroSwimmer,self).__init__()
        self.n_lookahead = n_lookahead
        self.action_space = gym.spaces.Box(
            shape=(2,),low = -np.inf,high = np.inf,dtype=np.float32
        )
        if velocity_bool:
            self.observation_space = gym.spaces.Box(
                shape=(2*(1+1+self.n_lookahead),),low = -np.inf,high = np.inf,dtype=np.float32
            )
        else : 
            self.observation_space = gym.spaces.Box(
                shape=(2*(1+self.n_lookahead),),low = -np.inf,high = np.inf,dtype=np.float32
            )

        self.x = x_0
        self.x_0=x_0
        self.previous_x= x_0
        self.C = C
        self.Dt = Dt
        self.U = 1
        self.dir_path=np.zeros(2)
        self.id_cp = 0
        self.d_cp = 0
        self.velocity_bool =velocity_bool
        self.seed = seed
        self.rng = np.random.default_rng(seed) 



    def state(self,tree,path):
        self.d,self.id_cp = min_dist_closest_point(self.x,tree)

        if self.id_cp<len(path)-1:
            self.dir_path = (path[self.id_cp+1]-path[self.id_cp])
        else:
            self.dir_path = (path[self.id_cp-1]-path[self.id_cp])
        p_cp = path[self.id_cp]
        s = coordinate_in_path_ref(p_cp,self.dir_path,self.x)
        s_previous = coordinate_in_path_ref(p_cp,self.dir_path,self.previous_x)
        v_local_path = (s- s_previous)/self.Dt
        if self.n_lookahead>0:
            lookahead = []
            for i in range(1, self.n_lookahead+1):
                idx = min(self.id_cp + i, len(path) - 1)
                lookahead.append(path[idx])
            lookahead_local = [coordinate_in_path_ref(p_cp, self.dir_path, p) for p in lookahead]
            if self.velocity_bool: 
                return np.concatenate((s.reshape(1,2),v_local_path.reshape(1,2),np.array(lookahead_local)),axis=0)
            else : 
                return np.concatenate((s.reshape(1,2).reshape(1,2),np.array(lookahead_local)),axis=0)
        if self.velocity_bool: 
            return np.concatenate((s.reshape(1,2),v_local_path.reshape(1,2)),axis=0)
        else : 
            return np.concatenate((s.reshape(1,2).reshape(1,2)),axis=0)

    
    def reward(self,x_target,beta):
        d =self.d
        rew_t = -self.C * self.Dt 
        rew_target=  - np.linalg.norm(self.x-x_target) +  np.linalg.norm(self.previous_x-x_target)
        rew_d = - beta*d
        rew =  rew_t + rew_d + rew_target
        return rew_t,rew_d,rew_target,rew
    
    def step(self,action,tree,path,x_target,beta,D=0.1,u_bg=np.zeros(2),threshold=0.2):
        rew_t,rew_d,rew_tar,rew= self.reward(x_target,beta)
        self.previous_x=self.x
        action_global_ref = coordinate_in_global_ref(np.zeros(2),self.dir_path,action)
        self.x = solver(x=self.x,U=self.U,p=action_global_ref,Dt=self.Dt,D=D,u_bg=u_bg,rng=self.rng)
        next_state = self.state(tree,path)
        done = False
        d = np.linalg.norm(self.x-x_target)
        if d<threshold:
            done = True
        return next_state,rew,done,{
            'x' :self.x,
            'rew_t': rew_t,
            'rew_d': rew_d,
            'rew_target': rew_tar,
        }
    
    def reset(self,tree,path): 
        self.x = self.x_0
        return self.state(tree,path)

        


