import numpy as np
from itertools import product


class GridWorld(object):
    basic_actions = [0, 1, 2, 3]
    basic_effects = {
        0: np.array([-1., 0.]),
        1: np.array([0., -1.]),
        2: np.array([1., 0.]),
        3: np.array([0., 1.])
    }
    basic_action_reps = {
        0: "↑",
        1: "←",
        2: "↓",
        3: "→",
    }

    def __init__(self, seq_len=2, size=5):
        self.seq_len = seq_len
        self.actions = np.arange(len(self.basic_actions) ** seq_len)
        self.all_actions = list(product(*[self.basic_actions for _ in range(seq_len)]))
        self.action_reps = {}
        self.action_effects = {}
        for i, v in enumerate(self.all_actions):
            rep = ""
            effect = np.zeros(2)
            for d in v:
                rep += self.basic_action_reps[d]
                effect += self.basic_effects[d]
            self.action_reps[i] = rep
            self.action_effects[i] = effect

        self.size = size
        self.grid = np.zeros((size, size), dtype=np.float)
        self.position = np.zeros(2, dtype=np.float)
        self.goal = np.zeros(2, dtype=np.float)

    def reset(self):
        self.position = np.random.random_integers(-self.size // 2, self.size // 2, 2).astype(np.float)
        self.goal = np.random.random_integers(-self.size, self.size, 2).astype(np.float)
        while all(self.goal == self.position):
            self.goal = np.random.random_integers(-self.size, self.size, 2).astype(np.float)
        # self.goal = np.array([self.size, self.size])

    def step_one(self, a):
        self.position += self.basic_effects[a]
        self.position[0] = self.size if self.position[0] > self.size else self.position[0]
        self.position[0] = -self.size if self.position[0] < -self.size else self.position[0]
        self.position[1] = self.size if self.position[1] > self.size else self.position[1]
        self.position[1] = -self.size if self.position[1] < -self.size else self.position[1]

        if self.is_terminal:
            return 10, True
        else:
            return -0.05, False

    def step(self, a, rand=False):
        actions = self.all_actions[a]
        reward = 0
        d = False
        for i, action in enumerate(actions):
            r, d = self.step_one(action)
            reward += r
            if d:
                return self.get_state(), reward, d, None
        return self.get_state(), reward, d, None

    def get_state(self):
        return np.concatenate([self.position, self.goal]) / self.size

    @property
    def is_terminal(self):
        if all(self.position == self.goal):
            return True
        return False


if __name__ == '__main__':
    grid = GridWorld(1)
    grid.reset()
    effects = np.array([grid.action_effects[i] for i in grid.actions])
    print(len(np.unique(effects, axis=0)))
    print(grid.goal, grid.position, grid.get_state())

