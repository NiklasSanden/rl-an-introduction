import numpy as np

def argmax(array):
    """
    The numpy argmax function breaks ties by choosing the first occurence. This implementation breaks ties uniformly at random.
    """
    assert (len(array.shape) == 1), "argmax expects a 1d array"
    assert (array.size > 0),        "argmax expects non-empty array"

    indices = []
    best = float('-inf')
    for i in range(len(array)):
        if array[i] > best:
            best = array[i]
            indices = [i]
        elif array[i] == best:
            indices.append(i)
    return np.random.choice(indices)

class GreedyActionSelector(object):
    def __call__(self, Q):
        return argmax(Q)

class EpsilonGreedyActionSelector(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self, Q):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, Q.size)
        else:
            return argmax(Q)
