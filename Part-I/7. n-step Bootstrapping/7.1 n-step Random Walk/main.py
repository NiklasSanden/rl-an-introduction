import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.agents import *

# PARAMETERS
NUM_RUNS = 100
EPISODES = 10
NUM_STATES = 19
ENVIRONMENT = RandomWalk(num_states=NUM_STATES)
AGENT = PredeterminedActions(ENVIRONMENT)
GAMMA = 1.0
TRUE_VALUES = [i / (NUM_STATES + 1) for i in range(-NUM_STATES + 1, NUM_STATES + 1, 2)]
ALPHAS = [x for x in np.linspace(0, 1, num=100)]
N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
COLOURS = ['red', 'green', 'blue', 'black', 'pink', 'lightblue', 'purple', 'magenta', 'orange', 'brown']

if __name__ == '__main__':
    fig = plt.figure()

    errors = np.zeros((NUM_RUNS, EPISODES, len(ALPHAS), len(N_VALUES)))
    for run in tqdm(range(NUM_RUNS)):
        for i in range(len(N_VALUES)):
            for j in range(len(ALPHAS)):
                AGENT.reset() # use the same set of actoins/walks for each parameter setting
                V = defaultdict(lambda: 0.0)
                for episode in range(EPISODES):
                    V = n_step_TD(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, n=N_VALUES[i], alpha=ALPHAS[j], start_V=V, log=False)
                    errors[run, episode, j, i] = RMS_error(V, NUM_STATES, TRUE_VALUES)
    
    errors = np.average(errors, axis=0)
    errors = np.average(errors, axis=0)
    
    ax = fig.add_subplot()
    for i in range(len(N_VALUES)):
        ax.plot(ALPHAS, errors[:, i], color=COLOURS[i], label='n=' + str(N_VALUES[i]))
    ax.set(ylabel='Average RMS error over ' + str(NUM_STATES) + ' states and first ' + str(EPISODES) + ' episodes', xlabel='alpha')
    ax.legend()

    plt.show()
