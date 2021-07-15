import numpy as np

from .misc import *

class Agent(object):
    def __init__(self, env):
        self.env = env

    def __call__(self, state, Q=None):
        raise NotImplementedError()
    
    def get_actions_and_probs(self, state, Q=None):
        '''
        Returns a pair with one list of all of the actions available in the state and an np-array 
        with the probability of each action being selected (in the same order as the first list).
        '''
        raise NotImplementedError()

    def get_probs_as_dict(self, state, Q=None):
        actions, probs = self.get_actions_and_probs(state, Q)
        return {actions[a]:probs[a] for a in actions}

class AgentWrapper(object):
    def __init__(self, agent):
        self.agent = agent
    
    def __call__(self, state, Q=None):
        return self.agent(state, Q)
    
    def get_actions_and_probs(self, state, Q=None):
        return self.agent.get_actions_and_probs(state, Q)

    def get_probs_as_dict(self, state, Q=None):
        return self.agent.get_probs_as_dict(state, Q)

    # Wrapper child functions
    def repeat(self):
        self.agent.repeat()
    
    def reset_actions(self):
        self.agent.reset_actions()

    def get_number_of_calls(self):
        return self.agent.get_number_of_calls()

    def reset_counter(self):
        self.agent.reset_counter()

class RepeatableWrapper(AgentWrapper):
    def __init__(self, agent):
        super().__init__(agent)
        self.reset_actions()

    def __call__(self, state, Q=None):
        if self.iterator < len(self.saved_actions):
            action = self.saved_actions[self.iterator]
        else:
            action = super().__call__(state, Q)
            self.saved_actions.append(action)
        self.iterator += 1
        return action

    def repeat(self):
        self.iterator = 0

    def reset_actions(self):
        self.iterator = 0
        self.saved_actions = []

class NumberOfActionsWrapper(AgentWrapper):
    '''
    This wrapper will keep track of the number of calls made to it which is used to 
    see how many time steps it took to complete each episode.
    '''
    def __init__(self, agent):
        super().__init__(agent)
        self.reset_counter()
    
    def __call__(self, state, Q=None):
        self.counter += 1
        return super().__call__(state, Q)

    def get_number_of_calls(self):
        return self.counter

    def reset_counter(self):
        self.counter = 0

class EpsilonGreedyAgent(Agent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon
    
    def __call__(self, state, Q):
        actions = self.env.get_actions(state)
        if np.random.rand() < self.epsilon:
            return actions[np.random.choice(np.arange(len(actions)))]
        else:
            best_index = argmax(np.array([Q(state, a) for a in actions]))
            return actions[best_index]
    
    def get_actions_and_probs(self, state, Q):
        '''
        Each tie of argmax share the 1.0 - epsilon probability.
        The argmax used in __call__ breaks ties randomly so this is correct.
        '''
        actions = self.env.get_actions(state)
        probs = np.ones(len(actions)) * self.epsilon / len(actions)
        Q_values = np.array([Q(state, a) for a in actions])
        best_indices = np.argwhere(Q_values == np.max(Q_values))
        probs[best_indices] += (1.0 - self.epsilon) / len(best_indices)
        return (actions, probs)