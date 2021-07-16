from .misc import *

class Environment(object):
    def get_actions(self, state):
        '''
        Returns the available actions for the state
        '''
        raise NotImplementedError()

    def step(self, action):
        '''
        Returns the observation of the next state, the reward, a terminal flag and a dictionary with debug information
        '''
        raise NotImplementedError()

    def render(self):
        pass

    def reset(self):
        '''
        Returns the initial observation of the starting state
        '''
        raise NotImplementedError()

class ShortCorridor(Environment):
    def __init__(self):
        self.reset()

    def get_actions(self, state):
        return [-1, 1]

    def step(self, action):
        if self.state == 1:
            action *= -1
        self.state = clip(self.state + action, 0, 3)
        terminal = self.state == 3
        return (self.state, -1, terminal, {})            

    def reset(self):
        self.state = 0
        return self.state
