import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.policy_evaluation import *
from utility.agents import *
from utility.function_approximators import *

# PARAMETERS
EPISODES = 100000
NUM_STATES = 1000
JUMP_SIZE = 100
ENVIRONMENT = RandomWalk(num_states=NUM_STATES, step_size=JUMP_SIZE)
AGENT = StateCounterWrapper(UniformAgent(ENVIRONMENT))
V = StateAggregation(num_states=NUM_STATES, num_bins=NUM_STATES // JUMP_SIZE, start_state=1)
MAX_DELTA = 0.0001
GAMMA = 1
ALPHA = 2 * 1e-5
WIDTH = 2.0
COLOURS = ['blue', 'red', 'gray']

if __name__ == '__main__':
    fig = plt.figure()

    true_values = ENVIRONMENT.get_true_value(GAMMA, MAX_DELTA)
    V = gradient_monte_carlo_v(ENVIRONMENT, AGENT, GAMMA, EPISODES, ALPHA, V, log=True)
    values = np.array([V(s) for s in range(1, NUM_STATES + 1)])

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(np.arange(1, NUM_STATES + 1), values, color=COLOURS[0], label='Approximate MC value v̂')
    ax.plot(np.arange(1, NUM_STATES + 1), true_values, color=COLOURS[1], label='True value v')
    ax.set(ylabel='Value', xlabel='State')
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    state_counters = np.array([AGENT.get_count(state) for state in range(1, NUM_STATES + 1)], dtype=float)
    state_counters /= np.sum(state_counters)
    ax.bar(np.arange(1, NUM_STATES + 1), state_counters, width=WIDTH, color=COLOURS[2], label='State distribution')
    ax.set(ylabel='Distribution', xlabel='State')
    ax.legend()

    plt.show()
