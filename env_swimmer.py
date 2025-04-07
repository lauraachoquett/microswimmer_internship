import gymnasium as gym
import numpy as np
from invariant_state import coordinate_in_path_ref
from distance_to_path import min_dist_closest_point
from sde import solver

class MicroSwimmer(gym.Env):
    def __init__(self,x_0,C,Dt,threshold):
        super(MicroSwimmer,self).__init__()

        self.action_space = gym.spaces.Box(
            shape=(2,),low = -np.inf,high = np.inf,dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            shape=(2,),low = -np.inf,high = np.inf,dtype=np.float32
        )

        self.x = x_0
        self.x_0=x_0
        self.previous_x= x_0
        self.C = C
        self.Dt = Dt
        self.U = 1
        self.D = 0.1
        self.threshold = threshold

    def state(self,p_0,T_0):
        s = coordinate_in_path_ref(p_0,T_0,self.x)
        return s
    
    def reward(self,path,x_target):
        d = min_dist_closest_point(self.x,path)
        rew_t = -self.C * self.Dt - np.linalg.norm(self.x-x_target) +  np.linalg.norm(self.previous_x-x_target)
        rew_d = - 0.05*d
        rew =  rew_t + rew_d
        ##print(f"reward temps : {rew_t}")
        #print(f"reward distance: {rew_d}")
        return rew
    
    def step(self,action,path,x_target,p_0,T_0):
        rew= self.reward(path,x_target)
        self.previous_x=self.x
        self.x = solver(x=self.x,U=self.U,p=action,Dt=self.Dt,D=self.D)
        next_state = self.state(p_0,T_0)
        done = False
        d = np.linalg.norm(self.x-x_target)
        if d<self.threshold:
            done = True
        return next_state,rew,done,{}
    
    def reset(self,p_0,T_0): 
        self.x = self.x_0
        return self.state(p_0,T_0)

        


