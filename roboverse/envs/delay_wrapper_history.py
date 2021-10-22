import gym
import numpy as np
import roboverse

class DelayWrapperHistory(gym.ObservationWrapper):
    def __init__(self, env, num_delay=1):
        super().__init__(env)
        self.prev_obs = [None] * num_delay
        self.num_delay = num_delay
        self._set_observation_space()
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_obs = [None] * self.num_delay
        return self.observation(observation)
    
    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.observation_img_dim ** 2) * 3 * (self.num_delay + 1)
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10 * (self.num_delay + 1)  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def observation(self, observation):
        self.prev_obs.append(observation)

        for i in range(len(self.prev_obs)):
            if self.prev_obs[i] is None:
                self.prev_obs[i] = {x:np.zeros_like(observation[x]) for x in observation} 
        
        out = {x:[] for x in self.prev_obs[0]}

        for i in range(len(self.prev_obs)):
            for x in self.prev_obs[i]:
                out[x].append(self.prev_obs[i][x])
        
        for key in out:
            out[key] = np.array(out[key]).flatten()

        self.prev_obs.pop(0)
        return out

if __name__ == "__main__":
    for i in range(1,3):
        print('delay', i)
        import roboverse
        env = DelayWrapperHistory(roboverse.make("Widow250DoubleDrawerCloseOpenGraspNeutral-v0"), num_delay=i)
        import ipdb;ipdb.set_trace()
        init_obs = env.reset()
        print(init_obs['image'])
        print(init_obs['image'].shape)
        for _ in range(i + 3):
            out = env.step(env.action_space.sample())
            print(out[0]['image'])
            print(out[0]['image'].shape)