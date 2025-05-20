import gymnasium as gym
import numpy as np
import torch

from src.distance_to_path import min_dist_closest_point
from src.invariant_state import *
from src.simulation import solver


class MicroSwimmer(gym.Env):
    def __init__(
        self,
        x_0,
        C,
        Dt,
        velocity_bool,
        n_lookahead=5,
        velocity_ahead=False,
        add_action=False,
        seed=None,
        bounce_thr=0,
        dim=2,
    ):
        
        super(MicroSwimmer, self).__init__()
        self.dim=dim
        self.n_lookahead = n_lookahead
        self.action_space = gym.spaces.Box(
            shape=(self.dim,), low=-np.inf, high=np.inf, dtype=np.float32
        )
        base = 1 # Position
        
        if add_action:
            base +=1 # Past_action
        if velocity_bool:
            base +=1 # Velocity of the agent
            
        if velocity_ahead:
                base +=2 * self.n_lookahead #Velocity and Position of points ahead
        else:
            base += self.n_lookahead  # Position of points ahead
           

        print("Base: ",base)

            
        self.observation_space = gym.spaces.Box(
            shape=(self.dim * base,),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

        self.x = x_0
        self.x_0 = x_0
        self.previous_x = x_0
        self.C = C
        self.Dt = Dt
        self.U = 1
        self.bounce_thr = bounce_thr
        self.dir_path = np.zeros(self.dim)
        self.t = np.zeros(self.dim)
        self.n = np.zeros(self.dim)
        self.b = np.zeros(self.dim)
        self.id_cp = 0
        self.d_cp = 0
        self.action = np.zeros(self.dim)
        self.velocity_ahead = velocity_ahead
        self.velocity_bool = velocity_bool
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        self.add_action = add_action

    def state(self, tree, path,T,N=None,B=None):
        if tree is None or path is None or T is None:
            return self.x

        self.d, self.id_cp = min_dist_closest_point(self.x, tree)
        path_len = len(path)
        p_cp = path[self.id_cp]
        
        if self.dim==2:
            self.dir_path = T[self.id_cp]
            s = coordinate_in_path_ref(p_cp, self.dir_path, self.x)
            s_previous = coordinate_in_path_ref(p_cp, self.dir_path, self.previous_x)

        else : 
            self.t = T[self.id_cp]
            self.n = N[self.id_cp]
            self.b = B[self.id_cp]
            s = coordinate_in_path_ref_3D(p_cp,self.x,self.t,self.n,self.b)
            s_previous = coordinate_in_path_ref_3D(p_cp,self.previous_x,self.t,self.n,self.b)

        v_local_path = (s - s_previous) / self.Dt

        result = [s.reshape(1, self.dim)]

        if self.velocity_bool:
            result.append(v_local_path.reshape(1, self.dim))

        if self.n_lookahead > 0:
            lookahead = []
            lookahead_vel = [] if self.velocity_ahead else None

            for i in range(1, self.n_lookahead + 1):
                idx = min(self.id_cp + i, path_len - 1)
                next_p = path[idx] 
                if self.dim==2:
                    self.dir_path = T[self.id_cp]
                    next_p_local=  coordinate_in_path_ref(p_cp, self.dir_path, next_p).reshape(1, self.dim)
                else : 
                    self.t = T[self.id_cp]
                    self.n = N[self.id_cp]
                    self.b = B[self.id_cp]
                    next_p_local = coordinate_in_path_ref_3D(p_cp,next_p,self.t,self.n,self.b).reshape(1, self.dim)
                   
                lookahead.append(
                   next_p_local
                )
                if self.velocity_ahead:
                    vel = self.velocity_func(next_p)
                    lookahead_vel.append(vel.reshape(1, self.dim))
            result.append(np.concatenate(lookahead, axis=0))
            if self.velocity_ahead:
                result.append(np.concatenate(lookahead_vel, axis=0))
        if self.add_action:
            result.append(self.action.reshape(1, self.dim))
        return np.concatenate(result, axis=0)

    def reward(self, x_target, beta):
        d = self.d
        rew_t = -self.C * self.Dt
        rew_target = -np.linalg.norm(self.x - x_target) + np.linalg.norm(
            self.previous_x - x_target
        )
        rew_d = -beta * d
        rew = rew_t + rew_d + rew_target
        return rew_t, rew_d, rew_target, rew

    def step(
        self,
        action,
        tree,
        path,
        T,
        x_target,
        beta,
        D=0.1,
        u_bg=None,
        threshold=0.2,
        sdf = None,
        N = None,
        B = None
        
    ):
        rew_t, rew_d, rew_tar, rew = self.reward(x_target, beta)
        self.previous_x = self.x
        self.action = action
        if self.dim==2:
            action_global_ref=  coordinate_in_global_ref(np.zeros(self.dim), self.dir_path, action)
        else : 
            action_global_ref= coordinate_in_global_ref_3D(np.zeros(self.dim),action,self.t,self.n,self.b)
        if u_bg is None: 
            u_bg = np.zeros(self.dim)
        self.x = solver(
            x=self.x,
            U=self.U,
            p=action_global_ref,
            Dt=self.Dt,
            D=D,
            u_bg=u_bg,
            bounce_thr=self.bounce_thr,
            sdf=sdf,
            dim=self.dim,
        )
        next_state = self.state(tree, path,T,N,B)
        done = False
        d = np.linalg.norm(self.x - x_target)
        if d < threshold:
            done = True
        return (
            next_state,
            rew,
            done,
            {
                "x": self.x,
                "rew_t": rew_t,
                "rew_d": rew_d,
                "rew_target": rew_tar,
            },
        )

    def reset(self, tree=None, path=None, T=None,velocity_func=None,N=None,B=None):
        if velocity_func is None:
            self.velocity_func = lambda x: np.zeros(self.dim)
        else:
            self.velocity_func = velocity_func
        self.x = self.x_0
        return self.state(tree, path,T,N,B)
