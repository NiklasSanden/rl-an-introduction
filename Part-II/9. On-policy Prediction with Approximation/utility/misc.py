import numpy as np
from tqdm import tqdm

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

def clip(x, a, b):
    '''
    Using numpy.clip on scalars is significantly slower than this.
    '''
    return min(b, max(x, a))

def calculate_gammas(gamma, highest_power):
    gammas = np.ones(highest_power + 1)
    for i in range(1, len(gammas)):
        gammas[i] = gammas[i - 1] * gamma
    return gammas

def get_state_distribution(env, agent, episodes, states, log=True):
    for i in tqdm(range(episodes), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            state, _, terminal, _ = env.step(agent(state))
    state_counters = np.array([agent.get_count(state) for state in states], dtype=float)
    return state_counters / np.sum(state_counters)

def calculate_VE(V, V_hat, states, distribution):
    sum = 0.0
    for s, mu in zip(states, distribution):
        sum += mu * (V(s) - V_hat(s)) ** 2
    return sum

class CircularList(object):
    def __init__(self, size, value=None):
        self.list = [value] * size
        self.size = size
    
    def __getitem__(self, key):
        return self.list[key % self.size]

    def __setitem__(self, key, value):
        self.list[key % self.size] = value
    
    def __len__(self):
        return self.size
