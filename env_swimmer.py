import gymnasium as gym
import numpy as np
from invariant_state import *
from distance_to_path import min_dist_closest_point
from sde import solver

class MicroSwimmer(gym.Env):
    def __init__(self,x_0,C,Dt,threshold,beta):
        super(MicroSwimmer,self).__init__()
        self.n_lookahead = 5
        self.action_space = gym.spaces.Box(
            shape=(2,),low = -np.inf,high = np.inf,dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            shape=(2*(1+self.n_lookahead),),low = -np.inf,high = np.inf,dtype=np.float32
        )

        self.x = x_0
        self.x_0=x_0
        self.previous_x= x_0
        self.C = C
        self.Dt = Dt
        self.U = 1
        self.D = 0.1
        self.threshold = threshold
        self.dir_path=np.zeros(2)
        self.beta=beta
        


    def state(self,path):
        d,id_cp = min_dist_closest_point(self.x,path)

        if id_cp<len(path)-1:
            self.dir_path = (path[id_cp+1]-path[id_cp])
        else:
            self.dir_path = (path[id_cp-1]-path[id_cp])
        p_cp = path[id_cp]
        s = coordinate_in_path_ref(p_cp,self.dir_path,self.x)

        if self.n_lookahead>0:
            lookahead = []
            for i in range(1, self.n_lookahead+1):
                idx = min(id_cp + i, len(path) - 1)
                lookahead.append(path[idx])
            lookahead_local = [coordinate_in_path_ref(p_cp, self.dir_path, p) for p in lookahead]
            return np.concatenate((s.reshape(1,2),np.array(lookahead_local)),axis=0)
        else: 
            return s
    
    def reward(self,path,x_target):
        d,_ = min_dist_closest_point(self.x,path)
        rew_t = -self.C * self.Dt - np.linalg.norm(self.x-x_target) +  np.linalg.norm(self.previous_x-x_target)
        rew_d = - self.beta*d
        rew =  rew_t + rew_d
        ##print(f"reward temps : {rew_t}")
        #print(f"reward distance: {rew_d}")
        return rew
    
    def step(self,action,path,x_target):
        rew= self.reward(path,x_target)
        self.previous_x=self.x
        action_global_ref = coordinate_in_global_ref(np.zeros(2),self.dir_path,action)
        self.x = solver(x=self.x,U=self.U,p=action_global_ref,Dt=self.Dt,D=self.D)
        next_state = self.state(path)
        done = False
        d = np.linalg.norm(self.x-x_target)
        if d<self.threshold:
            done = True
        return next_state,self.x,rew,done,{}
    
    def reset(self,path): 
        self.x = self.x_0
        return self.state(path)

        


