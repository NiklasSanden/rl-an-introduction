import numpy as np

class GradientFunctionApproximator(object):
    def __init__(self, d, weights=None):
        if weights is None:
            self.weights = np.random.randn(d)
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
    def __init__(self, num_states, num_bins, start_state=1):
        super().__init__(num_bins, np.zeros(num_bins))
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
    