import numpy as np

from .misc import *

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
