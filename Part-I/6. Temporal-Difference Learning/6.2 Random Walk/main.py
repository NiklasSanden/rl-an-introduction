import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *

# PARAMETERS
NUM_STATES = 5
TERMINALS = { -1, NUM_STATES }
ENVIRONMENT = RandomWalk(num_states=NUM_STATES)
AGENT = lambda s: np.random.choice(ENVIRONMENT.get_actions(s))
GAMMA = 1.0
TRUE_VALUES = [i / (NUM_STATES + 1) for i in range(1, NUM_STATES + 1)]
# TD_ONE_RUN PARAMETERS
TD_ONE_RUN_EPISODES = [0, 1, 10, 100]
TD_ONE_RUN_ALPHA = 0.1
TD_ONE_RUN_COLOURS = ['gray', 'red', 'green', 'blue']
# TD(0) MC COMPARISON PARAMETERS
NUM_RUNS = 100
EPISODES = 100
TD_ALPHAS = [0.05, 0.1, 0.15]
MC_ALPHAS = [0.01, 0.02, 0.03, 0.04]
TD_COLOUR = 'blue'
MC_COLOUR = 'red'
LINE_STYLES = ['solid', 'dotted', 'dashed', 'dashdot']

if __name__ == '__main__':
    fig = plt.figure()

    # TD One Run
    ax = fig.add_subplot(1, 2, 1)
    X = [i for i in range(1, NUM_STATES + 1)]
    for i in tqdm(range(len(TD_ONE_RUN_EPISODES))):
        V = TD0_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=TD_ONE_RUN_EPISODES[i], alpha=TD_ONE_RUN_ALPHA, 
                             start_V=defaultdict(lambda: 0.5), terminals=TERMINALS, log=False)
        v_array = [V[s] for s in range(NUM_STATES)]
        ax.plot(X, v_array, marker='o', color=TD_ONE_RUN_COLOURS[i], label=str(TD_ONE_RUN_EPISODES[i]))
    ax.plot(X, TRUE_VALUES, marker='o', color='black', label='True values')
    ax.set(title='One run of TD(0)', ylabel='Estimated value', xlabel='State')
    ax.legend()

    # TD MC Comparison
    ax = fig.add_subplot(1, 2, 2)
    TD_errors = np.zeros((NUM_RUNS, EPISODES, len(TD_ALPHAS)))
    MC_errors = np.zeros((NUM_RUNS, EPISODES, len(MC_ALPHAS)))
    X = np.arange(1, EPISODES + 1)
    for run in tqdm(range(NUM_RUNS)):
        
        for i in range(len(TD_ALPHAS)):
            TD_V = defaultdict(lambda: 0.5)
            for episode in range(EPISODES):
                TD_V = TD0_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=TD_ALPHAS[i], start_V=TD_V, terminals=TERMINALS, log=False)
                TD_errors[run, episode, i] = RMS_error(TD_V, NUM_STATES, TRUE_VALUES)
        
        for i in range(len(MC_ALPHAS)):
            MC_V = defaultdict(lambda: 0.5)
            for episode in range(EPISODES):
                MC_V = monte_carlo_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=MC_ALPHAS[i], start_V=MC_V, log=False)
                MC_errors[run, episode, i] = RMS_error(MC_V, NUM_STATES, TRUE_VALUES)
    
    TD_errors = np.average(TD_errors, axis=0)
    MC_errors = np.average(MC_errors, axis=0)
    for i in range(len(TD_ALPHAS)):
        ax.plot(X, TD_errors[:, i], color=TD_COLOUR, label='TD alpha=' + str(TD_ALPHAS[i]), linestyle=LINE_STYLES[i])
    for i in range(len(MC_ALPHAS)):
        ax.plot(X, MC_errors[:, i], color=MC_COLOUR, label='MC alpha=' + str(MC_ALPHAS[i]), linestyle=LINE_STYLES[i])
    ax.set(title='Comparison of TD(0) and MC', ylabel='Empirical RMS error, averaged over states', xlabel='Episodes')
    ax.legend()

    plt.show()
