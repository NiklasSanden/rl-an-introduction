import numpy as np

from .misc import *

class GradientFunctionApproximator(object):
    '''
    This generalizes the state value function by passing in the same action (0) every time.
    '''
    def __init__(self, d, weights=None):
        self.d = d
        if weights is None:
            self.weights = np.zeros(d)
        else:
            self.weights = weights
    
    def __call__(self, input, action=0):
        raise NotImplementedError()

    def get_target(self, weights, input, action=0):
        old_weights = self.weights
        self.weights = weights
        return_value = self.__call__(input, action)
        self.weights = old_weights
        return return_value

    def update_weights(self, alpha, change):
        self.weights = self.weights + alpha * change

    def get_gradients(self, input, action=0):
        raise NotImplementedError()

    def zero_weights(self):
        self.weights = np.zeros_like(self.weights)

class Tabular(GradientFunctionApproximator):
    '''
    This is a tabular setting using the framework of function approximation (see exercise 9.1)
    '''
    def __init__(self, num_states, num_actions=1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.map_to_weight = dict()
        self.index_iterator = 0
        super().__init__(num_states * num_actions)

    def __call__(self, input, action=0):
        return self.weights[self._get_index(input, action)]

    def get_gradients(self, input, action=0):
        # This could also just use indices (like __call__), but the current code for updating the weights would not support it.
        index = self._get_index(input, action)
        gradient = np.zeros(self.d)
        gradient[index] = 1.0
        return gradient

    def _get_index(self, input, action=0):
        if not ((input, action) in self.map_to_weight):
            self.map_to_weight[(input, action)] = self.index_iterator
            self.index_iterator += 1
        return self.map_to_weight[(input, action)]
