import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *
from utility.function_approximators import *

# PARAMETERS
EPISODES = 100000
NUM_STATES = 1000
JUMP_SIZE = 100
ENVIRONMENT = RandomWalk(num_states=NUM_STATES, step_size=JUMP_SIZE)
AGENT = UniformAgent(ENVIRONMENT)
V = StateAggregation(num_states=NUM_STATES, num_bins=NUM_STATES // JUMP_SIZE, start_state=1)
MAX_DELTA = 0.0001
GAMMA = 1
ALPHA = 0.0005
COLOURS = ['blue', 'red']

if __name__ == '__main__':
    fig = plt.figure()

    true_values = ENVIRONMENT.get_true_value(GAMMA, MAX_DELTA)
    V = semi_gradient_TD_0_v(ENVIRONMENT, AGENT, GAMMA, EPISODES, ALPHA, V, log=True)
    values = np.array([V(s) for s in range(1, NUM_STATES + 1)])

    ax = fig.add_subplot()
    ax.plot(np.arange(1, NUM_STATES + 1), values, color=COLOURS[0], label='Approximate TD value v̂')
    ax.plot(np.arange(1, NUM_STATES + 1), true_values, color=COLOURS[1], label='True value v')
    ax.set(ylabel='Value', xlabel='State')
    ax.legend()

    plt.show()
