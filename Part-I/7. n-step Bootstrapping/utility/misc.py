import numpy as np

import math

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

def RMS_error(V, num_states, true_values):
    error = 0
    for s in range(num_states):
        error += (V[s] - true_values[s]) ** 2
    error /= num_states
    return math.sqrt(error)
