import gym, roboschool
import numpy as np
from itertools import product


class MujocoWrapper(object):
    def __init__(self, env_name, action_num=101):
        self.env = gym.make(env_name)
        self.min_action = self.action_space.low
        self.max_action = self.action_space.high

        # self.real_actions = np.linspace(self.min_action, self.max_action, action_num) #.reshape(action_num, -1)
        real_actions = product(*[np.linspace(self.min_action[i], self.max_action[i], action_num) for i in range(len(self.min_action))])
        self.real_actions = np.array(list(real_actions))
        self.action_num = len(self.real_actions)
        self.actions = np.arange(self.action_num)
        self.obs = None
        self.reward = 0
        self.env.reset()

    def get_state(self):
        return self.obs.copy()

    def reset(self):
        self.obs = self.env.reset()
        return self.get_state()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, a):
        a = self.real_actions[a]
        self.obs, r, done, info = self.env.step(a)
        return self.get_state(), r, done, info

    def render(self):
        self.env.render()


if __name__ == '__main__':
    env = MujocoWrapper('Reacher-v2')
    print(env.step(0))
    for i in range(10):
        print(env.step(5))
    print(env.actions)
