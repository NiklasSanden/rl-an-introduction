import numpy as np

from .misc import *

from collections import defaultdict

class Agent(object):
    def __init__(self, env):
        self.env = env

    def __call__(self, state):
        raise NotImplementedError()
    
    def get_actions_and_probs(self, state):
        '''
        Returns a pair with one list of all of the actions available in the state and an np-array 
        with the probability of each action being selected (in the same order as the first list).
        '''
        raise NotImplementedError()

    def get_probs_as_dict(self, state):
        actions, probs = self.get_actions_and_probs(state)
        return {actions[a]:probs[a] for a in actions}

class StateCounterWrapper(object):
    def __init__(self, agent):
        self.agent = agent
        self.counter = defaultdict(lambda: 0)

    def get_count(self, state):
        return self.counter[state]
    
    def __call__(self, state):
        self.counter[state] += 1
        return self.agent(state)
    
    def get_actions_and_probs(self, state):
        return self.agent.get_actions_and_probs(state)

    def get_probs_as_dict(self, state):
        return self.agent.get_probs_as_dict(state)

class UniformAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
    
    def __call__(self, state):
        actions = self.env.get_actions(state)
        return actions[np.random.choice(np.arange(len(actions)))]

    def get_actions_and_probs(self, state):
        actions = self.env.get_actions(state)
        probs = np.ones(len(actions)) / len(actions)
        return (actions, probs)
