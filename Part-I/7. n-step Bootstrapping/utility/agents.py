import numpy as np

from .misc import *

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

class EpsilonGreedyAgent(object):
    '''
    This agent will keep track of the number of calls made to it which is used to see how many time steps it took to complete
    each episode.
    '''
    def __init__(self, env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.counter = 0
    
    def __call__(self, state, Q):
        self.counter += 1
        actions = self.env.get_actions(state)
        if np.random.rand() < self.epsilon:
            return actions[np.random.choice(np.arange(len(actions)))]
        else:
            best_index = argmax(np.array([Q[(state, a)] for a in actions]))
            return actions[best_index]
    
    def get_probs(self, state, Q):
        '''
        Each tie of argmax share the 1.0 - epsilon probability.
        The argmax used in __call__ breaks ties randomly so this is correct.
        '''
        actions = self.env.get_actions(state)
        probs = np.ones(len(actions)) * self.epsilon / len(actions)
        Q_values = np.array([Q[(state, a)] for a in actions])
        best_indices = np.argwhere(Q_values == np.max(Q_values))
        probs[best_indices] += (1.0 - self.epsilon) / len(best_indices)
        return (actions, probs)

class GreedyAgent(object):
    def __init__(self, env):
        self.env = env
        self.counter = 0

    def __call__(self, state, Q):
        self.counter += 1
        actions = self.env.get_actions(state)
        best_index = argmax(np.array([Q[(state, a)] for a in actions]))
        return actions[best_index]
    
    def get_probs(self, state, Q):
        '''
        Each tie of argmax share the 1.0 - epsilon probability.
        The argmax used in __call__ breaks ties randomly so this is correct.
        '''
        actions = self.env.get_actions(state)
        probs = np.zeros(len(actions))
        Q_values = np.array([Q[(state, a)] for a in actions])
        best_indices = np.argwhere(Q_values == np.max(Q_values))
        probs[best_indices] += 1 / len(best_indices)
        return (actions, probs)
