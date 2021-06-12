import numpy as np
import math

class Environment(object):
    def __init__(self):
        pass

    def get_actions(self, state):
        '''
        Returns the available actions for the state
        '''
        NotImplementedError()

    def step(self, action):
        '''
        Returns the observation of the next state, the reward, a terminal flag and a dictionary with debug information
        '''
        NotImplementedError()

    def render(self):
        pass

    def reset(self):
        '''
        Returns the initial observation of the starting state
        '''
        NotImplementedError()

class RandomWalk(Environment):
    def __init__(self, num_states=5):
        '''
        num_states is excluding the two terminal states on the left and right
        '''
        assert num_states >= 1 and (num_states % 2) == 1, str(num_states) + ' is not a valid num_states'
        self.num_states = num_states
        self.reset()

    def get_actions(self, state):
        return [-1, 1]
    
    def step(self, action):
        self.state += action
        reward = 0.0
        terminal = False
        if self.state < 0:
            terminal = True
        elif self.state >= self.num_states:
            terminal = True
            reward = 1.0
        return (self.state, reward, terminal, {})

    def reset(self):
        self.state = self.num_states // 2
        return self.state