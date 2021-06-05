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

def get_most_likely_action(PI, s):
    best_prob = 0.0
    best_action = -1
    for a in PI(s):
        if PI(s)[a] > best_prob:
            best_prob = PI(s)[a]
            best_action = a
    return best_action
