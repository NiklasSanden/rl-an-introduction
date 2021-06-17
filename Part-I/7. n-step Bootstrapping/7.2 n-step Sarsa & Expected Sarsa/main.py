import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.agents import *
from utility.control_algorithms import *

# PARAMETERS
NUM_RUNS = 500
EPISODES = 200
ENVIRONMENT = WindyGridworld(rows=7, cols=10, wind_values=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0], is_stochastic=False, king_moves=False, can_pass=False)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
N_VALUES = [1, 4, 8]
ALGORITHMS = [n_step_sarsa, n_step_expected_sarsa_on_policy]
GAMMA = 1.0
ALPHAS = [0.5, 0.2, 0.1] # this is sort of akward but they want very different learning rates for such few episodes and it impacts
                         # the results here (look at 7.1 Sum of TD-errors Comparison). Note that this is a different environment etc.
COLOURS = ['red', 'green', 'blue'] # each n_value has a different colour
LINESTYLES = ['solid', 'dashed']   # and each algorithm a different linestyle
ALGORITHM_LABELS = ['sarsa', 'expected sarsa']

if __name__ == '__main__':
    fig = plt.figure()

    episode_length = np.zeros((NUM_RUNS, EPISODES, len(N_VALUES), len(ALGORITHMS)))
    for run in tqdm(range(NUM_RUNS)):
        for n in range(len(N_VALUES)):
            for a in range(len(ALGORITHMS)):
                Q = defaultdict(lambda: 0.0)
                for episode in range(EPISODES):
                    AGENT.counter = 0
                    Q = ALGORITHMS[a](ENVIRONMENT, AGENT, GAMMA, max_iterations=1, n=N_VALUES[n], alpha=ALPHAS[n], start_Q=Q, log=False)
                    episode_length[run, episode, n, a] = AGENT.counter
    
    episode_length = np.average(episode_length, axis=0)
    
    ax = fig.add_subplot()
    for a in range(len(ALGORITHMS)):
        for n in range(len(N_VALUES)):
            ax.plot(episode_length[:, n, a], linestyle=LINESTYLES[a], color=COLOURS[n], label=ALGORITHM_LABELS[a] + ' n=' + str(N_VALUES[n]))
    ax.set(ylabel='Episode length averaged over' + str(NUM_RUNS) + ' runs (log scale)', xlabel='Episode')
    ax.set_yscale('log', base=10)
    ax.legend()

    plt.show()
