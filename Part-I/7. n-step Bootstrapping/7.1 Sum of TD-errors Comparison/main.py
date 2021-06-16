import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.agents import *

# Change n_step_TD to n_step_sum_of_TD_errors with use_latest_V_for_TD to True and compare with n_step_TD to
# show empirically what is proven in exercise 7.1.
# PARAMETERS
NUM_RUNS = 100
EPISODES = 10
NUM_STATES = 19
ENVIRONMENT = RandomWalk(num_states=NUM_STATES)
AGENT = PredeterminedActions(ENVIRONMENT)
GAMMA = 1.0
TRUE_VALUES = [i / (NUM_STATES + 1) for i in range(-NUM_STATES + 1, NUM_STATES + 1, 2)]
ALPHAS = [x for x in np.linspace(0, 1, num=50)]
N_VALUES = [1, 2, 4, 8, 16, 32]
COLOURS = ['red', 'green', 'blue', 'black', 'pink', 'lightblue']
N_STEP_LINESTYLE = 'dashed'
TD_SUM_LINESTYLE = 'solid'

if __name__ == '__main__':
    fig = plt.figure()

    n_step_errors = np.zeros((NUM_RUNS, EPISODES, len(ALPHAS), len(N_VALUES)))
    td_sum_errors = np.zeros((NUM_RUNS, EPISODES, len(ALPHAS), len(N_VALUES)))
    for run in tqdm(range(NUM_RUNS)):
        for i in range(len(N_VALUES)):
            for j in range(len(ALPHAS)):
                AGENT.reset() # use the same set of actions/walks for each parameter setting
                V = defaultdict(lambda: 0.0)
                for episode in range(EPISODES):
                    V = n_step_TD(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, n=N_VALUES[i], alpha=ALPHAS[j], start_V=V, log=False)
                    n_step_errors[run, episode, j, i] = RMS_error(V, NUM_STATES, TRUE_VALUES)

                AGENT.reset() # use the same set of actions/walks for each parameter setting
                V = defaultdict(lambda: 0.0)
                for episode in range(EPISODES):
                    V = n_step_sum_of_TD_errors_iterative(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, n=N_VALUES[i], alpha=ALPHAS[j], start_V=V, log=False)
                    td_sum_errors[run, episode, j, i] = RMS_error(V, NUM_STATES, TRUE_VALUES)
    
    n_step_errors = np.average(n_step_errors, axis=(0, 1))
    td_sum_errors = np.average(td_sum_errors, axis=(0, 1))
    
    ax = fig.add_subplot()
    for i in range(len(N_VALUES)): # the loops are separated to order it nicely in the legend
        ax.plot(ALPHAS, n_step_errors[:, i], linestyle=N_STEP_LINESTYLE, color=COLOURS[i], label='n_step n=' + str(N_VALUES[i]))
    for i in range(len(N_VALUES)):
        ax.plot(ALPHAS, td_sum_errors[:, i], linestyle=TD_SUM_LINESTYLE, color=COLOURS[i], label='td_sum n=' + str(N_VALUES[i]))
    ax.set(ylabel='Average RMS error over ' + str(NUM_STATES) + ' states and first ' + str(EPISODES) + ' episodes', xlabel='alpha')
    ax.set_ylim([0.2, 0.8]) # you might wanna zoom out/in after this but some diverge so it is necessary.
    ax.legend()

    plt.show()
