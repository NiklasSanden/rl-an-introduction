import numpy as np

def argmax(array):
    '''
    The numpy argmax function breaks ties by choosing the first occurence. This implementation breaks ties uniformly at random.
    '''
    assert (len(array.shape) == 1), 'argmax expects a 1d array'
    assert (array.size > 0),        'argmax expects non-empty array'

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
    
    def reset(self):
        pass

class EpsilonGreedyActionSelector(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self, Q):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, Q.size)
        else:
            return argmax(Q)
    
    def reset(self):
        pass

class UCBActionSelector(object):
    def __init__(self, k, c):
        self.k = k
        self.c = c
        self.reset()

    def __call__(self, Q):
        array = Q + self.c * np.sqrt(np.log(self.t) / np.maximum(1, self.N))
        array = (self.N > 0) * array + (self.N == 0) * np.ones(self.k) * 1000000000.0
        action = argmax(array)
        
        self.N[action] += 1
        self.t += 1
        return action

    def reset(self):
        self.N = np.zeros(self.k, dtype=int)
        self.t = 1

class StochasticActionSelector(object):
    def __call__(self, pi):
        return np.random.choice(range(pi.size), p=pi)

    def reset(self):
        pass
