import numpy as np
import matplotlib.pyplot as plt

import math

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.policy_evaluation import *
from utility.agents import *
from utility.function_approximators import *

# PARAMETERS
NUM_RUNS = 100
EPISODES = 10
NUM_STATES = 1000
JUMP_SIZE = 100
ENVIRONMENT = RandomWalk(num_states=NUM_STATES, step_size=JUMP_SIZE)
AGENT = RepeatableWrapper(UniformAgent(ENVIRONMENT))
V = StateAggregation(num_states=NUM_STATES, num_bins=20, start_state=1)
GAMMA = 1.0
TRUE_VALUES = [i / (NUM_STATES + 1) for i in range(-NUM_STATES + 1, NUM_STATES + 1, 2)]
ALPHAS = [x for x in np.linspace(0, 1, num=50)]
N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
COLOURS = ['red', 'green', 'blue', 'black', 'pink', 'lightblue', 'purple', 'magenta', 'orange', 'brown']

def RMS_error(V, num_states, true_values):
    error = 0
    for s in range(num_states):
        error += (V(s) - true_values[s]) ** 2
    error /= num_states
    return math.sqrt(error)

if __name__ == '__main__':
    fig = plt.figure()

    errors = np.zeros((NUM_RUNS, EPISODES, len(ALPHAS), len(N_VALUES)))
    for run in tqdm(range(NUM_RUNS)):
        AGENT.reset() # use different actions for each run
        for i in tqdm(range(len(N_VALUES)), leave=False):
            for j in range(len(ALPHAS)):
                V.zero_weights()
                AGENT.repeat() # ensure that the actions are the same for all hyperparameters
                for episode in range(EPISODES):
                    V = semi_gradient_n_step_TD_v(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, alpha=ALPHAS[j], n=N_VALUES[i], V=V, log=False)
                    errors[run, episode, j, i] = RMS_error(V, NUM_STATES, TRUE_VALUES)
    
    errors = np.average(errors, axis=(0, 1))
    
    ax = fig.add_subplot()
    for i in range(len(N_VALUES)):
        ax.plot(ALPHAS, errors[:, i], color=COLOURS[i], label='n=' + str(N_VALUES[i]))
    ax.set(ylabel='Average RMS error over ' + str(NUM_STATES) + ' states and first ' + str(EPISODES) + ' episodes', xlabel='alpha')
    ax.legend()

    plt.show()
