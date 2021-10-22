import gym
import numpy as np

class DelayWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_delay=1):
        super().__init__(env)
        self.prev_obs = [None] * num_delay
        self.num_delay = num_delay
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_obs = [None] * self.num_delay
        return self.observation(observation)

    def observation(self, observation):
        self.prev_obs.append(observation)
        obs = self.prev_obs.pop(0)
        return {x:np.zeros_like(observation[x]) for x in observation} if obs is None else obs
    
if __name__ == "__main__":
    for i in range(2,4):
        print('delay', i)
        import roboverse
        env = DelayWrapper(roboverse.make("Widow250DoubleDrawerCloseOpenGraspNeutral-v0"), num_delay=i)
        print(env.reset()['image'])
        for _ in range(i + 3):
            out = env.step(env.action_space.sample())
            print(out[0]['image'])