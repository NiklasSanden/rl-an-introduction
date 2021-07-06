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

class AgentWrapper(object):
    def __init__(self, agent):
        self.agent = agent
    
    def __call__(self, state):
        return self.agent(state)
    
    def get_actions_and_probs(self, state):
        return self.agent.get_actions_and_probs(state)

    def get_probs_as_dict(self, state):
        return self.agent.get_probs_as_dict(state)

    # Wrapper child functions
    def get_count(self, state):
        return self.agent.get_count(state)

    def repeat(self):
        self.agent.repeat()

    def reset(self):
        self.agent.reset()

class StateCounterWrapper(AgentWrapper):
    def __init__(self, agent):
        super().__init__(agent)
        self.counter = defaultdict(lambda: 0)

    def get_count(self, state):
        return self.counter[state]
    
    def __call__(self, state):
        self.counter[state] += 1
        return super().__call__(state)
    
class RepeatableWrapper(AgentWrapper):
    def __init__(self, agent):
        super().__init__(agent)
        self.reset()

    def __call__(self, state):
        if self.iterator < len(self.saved_actions):
            action = self.saved_actions[self.iterator]
        else:
            action = super().__call__(state)
            self.saved_actions.append(action)
        self.iterator += 1
        return action

    def repeat(self):
        self.iterator = 0

    def reset(self):
        self.iterator = 0
        self.saved_actions = []

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
