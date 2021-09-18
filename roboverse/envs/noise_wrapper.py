import gym
import numpy as np

class StochasticDynamicsWrapper(gym.ActionWrapper):
    def __init__(self, env, std=0.1, seed=0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.std = std

    def action(self, act):
        act = act + self.rng.normal(0, self.std, act.shape)
        return act