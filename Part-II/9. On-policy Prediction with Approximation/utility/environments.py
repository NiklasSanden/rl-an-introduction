import numpy as np

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

class RandomWalk(Environment):
    def __init__(self, num_states=1000, step_size=100):
        '''
        num_states is excluding the two terminal states on the left and right
        '''
        assert num_states >= 1, str(num_states) + ' is not a valid num_states'
        self.num_states = num_states
        self.step_size = step_size
        self.reset()

    def get_actions(self, state):
        return [-x for x in range(1, self.step_size + 1)] + [x for x in range(1, self.step_size + 1)]
    
    def step(self, action):
        self.state += action
        self.state = max(min(self.state, self.num_states + 1), 0)
        reward = 0.0
        terminal = False
        if self.state <= 0:
            terminal = True
            reward = -1.0
        elif self.state > self.num_states:
            terminal = True
            reward = 1.0
        return (self.state, reward, terminal, {})

    def reset(self):
        self.state = self.num_states // 2
        return self.state
