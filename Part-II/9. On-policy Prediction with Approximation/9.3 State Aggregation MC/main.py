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
AGENT = StateCounterWrapper(UniformAgent(ENVIRONMENT))
V = StateAggregation(num_states=NUM_STATES, num_bins=NUM_STATES // JUMP_SIZE, start_state=1)
MAX_DELTA = 0.0001
GAMMA = 1
ALPHA = 2 * 1e-5
WIDTH = 2.0
COLOURS = ['blue', 'red', 'gray']

# This is highly dependant on the implementation of RandomWalk
def get_true_value():
    optimal_V = np.zeros(NUM_STATES)
    while True:
        delta = 0.0
        for state in range(NUM_STATES):
            actions = ENVIRONMENT.get_actions(state)
            old_est = optimal_V[state]
            sum = 0.0
            for action in actions:
                s_ = state + action
                value_s_ = optimal_V[s_] if s_ < NUM_STATES and s_ >= 0 else 0
                if s_ >= NUM_STATES:
                    r = 1
                elif s_ < 0:
                    r = -1
                else:
                    r = 0
                sum += 1 / len(actions) * (r + GAMMA * value_s_)
            optimal_V[state] = sum

            delta = max(delta, abs(old_est - optimal_V[state]))
        
        print('policy evaluation error:', delta)
        if delta <= MAX_DELTA:
            break
    
    return optimal_V

if __name__ == '__main__':
    fig = plt.figure()

    true_values = get_true_value()
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
