import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
NUM_RUNS = 500
B_VALUES = [2, 10, 100, 1000, 10000]
COLOURS = ['blue', 'green', 'yellow', 'red', 'magenta']

class NextQValues(object):
    def __init__(self, b):
        self.b = b
        self.Q = np.random.randn(b)
    
    def sample(self):
        return np.random.choice(self.Q)
    
    def expected_answer(self):
        return np.average(self.Q)

def RMSError(error):
    return abs(error)

def plot(ax, id):
    b = B_VALUES[id]
    errors = np.zeros((NUM_RUNS, 2 * b))
    for run in tqdm(range(NUM_RUNS), leave=False):
        values = NextQValues(b)
        true_value = values.expected_answer()
        estimate = 0
        for i in range(2 * b):
            estimate += 1 / (i + 1) * (values.sample() - estimate)
            errors[run, i] = RMSError(estimate - true_value)
    errors = np.average(errors, axis=0)
    X = np.arange(1, len(errors) + 1) / b
    ax.plot(X, errors, label='b=' + str(b), color=COLOURS[id])

if __name__ == '__main__':
    fig = plt.figure()

    ax = fig.add_subplot()
    for i in tqdm(range(len(B_VALUES))):
        plot(ax, i)
    ax.plot([0, 1, 1, 2], [1, 1, 0, 0], label='expected updates', color='gray')
    ax.set(ylabel='RMS error in value estimate', xlabel='Number of max Q(S\', A\') computations')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0', '1b', '2b'])
    ax.legend()

    plt.show()
