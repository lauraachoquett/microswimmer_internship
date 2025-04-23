import gymnasium as gym
import numpy as np


class MicroSwimmerWrapper(gym.Wrapper):
    def __init__(self, env, tree, path, x_target, beta, D=0.1, u_bg=None):
        super().__init__(env)
        self.tree = tree
        self.path = path
        self.x_target = x_target
        self.beta = beta
        self.D = D
        self.u_bg = np.zeros(2) if u_bg is None else u_bg

    def reset(self, **kwargs):
        return self.env.reset(self.tree, self.path)

    def step(self, action):
        return self.env.step(
            action, self.tree, self.path, self.x_target, self.beta, self.D, self.u_bg
        )
