import numpy as np
import math

from .misc import *

class GradientFunctionApproximator(object):
    def __init__(self, d, weights=None):
        if weights is None:
            self.weights = np.zeros(d)
        else:
            self.weights = weights
    
    def __call__(self, input):
        raise NotImplementedError()

    def get_target(self, input, weights=None):
        old_weights = self.weights
        self.weights = weights
        return_value = self.__call__(input)
        self.weights = old_weights
        return return_value

    def update_weights(self, alpha, change):
        self.weights = self.weights + alpha * change

    def get_gradients(self, input):
        raise NotImplementedError()

    def zero_weights(self):
        self.weights = np.zeros_like(self.weights)

class StateAggregation(GradientFunctionApproximator):
    '''
    Expects 1d input
    '''
    def __init__(self, num_states, num_bins, start_state=1):
        super().__init__(num_bins)
        self.num_states = num_states
        self.bin_size = num_states / num_bins
        self.start_state = start_state

    def __call__(self, input):
        bin = int((input - self.start_state) / self.bin_size)
        return self.weights[bin]

    def get_gradients(self, input):
        bin = int((input - self.start_state) / self.bin_size)
        gradient = np.zeros_like(self.weights)
        gradient[bin] = 1.0
        return gradient

class PolynomialBasis(GradientFunctionApproximator):
    '''
    Expects 1d input
    '''
    def __init__(self, n, highest_value, lowest_value=0):
        self.d = n + 1
        self.lowest_value = lowest_value
        self.interval = highest_value - lowest_value
        super().__init__(self.d)

    def __call__(self, input):
        feature_vector = self._construct_feature_vector(input)
        return np.dot(feature_vector, self.weights)

    def get_gradients(self, input):
        return self._construct_feature_vector(input)
    
    def _construct_feature_vector(self, input):
        input -= self.lowest_value
        input /= self.interval
        input = clip(input, 0, 1)
        powers = np.ones(self.d)
        for i in range(1, len(powers)):
            powers[i] = powers[i - 1] * input
        return powers

class FourierBasis(GradientFunctionApproximator):
    '''
    Expects 1d input
    '''
    def __init__(self, n, highest_value, lowest_value=0):
        self.d = n + 1
        self.lowest_value = lowest_value
        self.interval = highest_value - lowest_value
        super().__init__(self.d)

    def __call__(self, input):
        feature_vector = self._construct_feature_vector(input)
        return np.dot(feature_vector, self.weights)

    def get_gradients(self, input):
        return self._construct_feature_vector(input)
    
    def _construct_feature_vector(self, input):
        input -= self.lowest_value
        input /= self.interval
        input = clip(input, 0, 1)
        return np.cos(math.pi * input * np.arange(self.d))

class TileCoding(GradientFunctionApproximator):
    '''
    Expects 1d input
    '''
    def __init__(self, num_tilings, width, highest_value, lowest_value=0):
        self.num_tilings = num_tilings
        self.width = width
        self.lowest_value = lowest_value
        interval = highest_value - lowest_value
        self.max_tiles = int(math.ceil((interval + (num_tilings - 1) * (width / num_tilings)) / width))
        self.d = self.max_tiles * num_tilings
        super().__init__(self.d)

    def __call__(self, input):
        indices = self._get_indices(input)
        return np.sum([self.weights[i] for i in indices])

    def get_gradients(self, input):
        # This could also just use indices (like __call__), but the current code for updating the weights would not support it.
        indices = self._get_indices(input)
        gradient = np.zeros(self.d)
        gradient[indices] = 1.0
        return gradient

    def update_weights(self, alpha, change):
        '''
        The alpha to use in this type of tile coding is alpha / num_tilings.
        '''
        self.weights = self.weights + alpha / self.num_tilings * change
    
    def _get_indices(self, input):
        input -= self.lowest_value
        # The clip makes sure that the index is valid even if there are outliers not in the interval [lowest_value, highest_value]
        # Also, the very end point of the last interval/tile is not included normally, so a form of clip or min is necessary there 
        return [clip(int((input + i * (self.width / self.num_tilings)) / self.width), 0, self.max_tiles - 1) + i * self.max_tiles for i in range(self.num_tilings)]
