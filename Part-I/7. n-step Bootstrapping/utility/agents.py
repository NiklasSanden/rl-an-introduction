import numpy as np

class PredeterminedActions(object):
    def __init__(self, env, max_actions=1000000):
        self.max_actions = max_actions
        self.actions = np.random.choice(env.get_actions(0), size=max_actions, replace=True)
        self.reset()

    def __call__(self, s):
        action = self.actions[self.iterator]
        self.iterator = (self.iterator + 1) % self.max_actions
        return action

    def reset(self):
        self.iterator = 0