import numpy as np

from .misc import *
from .rich_tile_coding import *

class GradientFunctionApproximator(object):
    def __init__(self, d, weights=None):
        if weights is None:
            self.weights = np.zeros(d)
        else:
            self.weights = weights
    
    def __call__(self, input, action):
        raise NotImplementedError()

    def get_target(self, input, action, weights=None):
        old_weights = self.weights
        self.weights = weights
        return_value = self.__call__(input, action)
        self.weights = old_weights
        return return_value

    def update_weights(self, alpha, change):
        self.weights = self.weights + alpha * change

    def get_gradients(self, input, action):
        raise NotImplementedError()

    def zero_weights(self):
        self.weights = np.zeros_like(self.weights)

class TileCoding(GradientFunctionApproximator):
    '''
    Uses the tile coding software provided by Richard Sutton (see rich_tile_coding.py)
    '''
    def __init__(self, num_tilings, dim_sizes, max_size=4096):
        self.num_tilings = num_tilings
        self.dim_sizes = dim_sizes
        self.iht = IHT(max_size)
        self.d = max_size
        super().__init__(max_size)

    def __call__(self, input, action):
        indices = self._get_indices(input, action)
        return np.sum([self.weights[i] for i in indices])

    def get_gradients(self, input, action):
        # This could also just use indices (like __call__), but the current code for updating the weights would not support it.
        indices = self._get_indices(input, action)
        gradient = np.zeros(self.d)
        gradient[indices] = 1.0
        return gradient

    def update_weights(self, alpha, change):
        '''
        The alpha to use in this type of tile coding is alpha / num_tilings.
        '''
        self.weights = self.weights + alpha / self.num_tilings * change
    
    def _get_indices(self, input, action):
        return tiles(self.iht, self.num_tilings, [self.num_tilings * x / size for x, size in zip(input, self.dim_sizes)], [action])
