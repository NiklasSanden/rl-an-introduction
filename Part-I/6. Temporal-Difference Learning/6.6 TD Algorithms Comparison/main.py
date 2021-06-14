import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
ENVIRONMENT = CliffWalkingSumOfRewardsWrapper(rows=4, cols=12)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
GAMMA = 1.0
ALPHAS = np.linspace(0.1, 1.0, 19)
ALGORITHMS = [sarsa_on_policy_td_q, q_learning, expected_sarsa_on_policy]
# INTERIM PARAMETERS
NUM_RUNS = 1000
EPISODES = 100
LABELS = ['Interim Sarsa', 'Interim Q-learning', 'Interim Expected Sarsa']
COLOURS = ['blue', 'black', 'red']
MARKERS = ['v', 's', 'x']
LINESTYLES = ['dotted'] * 3
# ASYMPTOTIC PARAMETERS
A_NUM_RUNS = 10
A_EPISODES = 10000
A_LABELS = ['Asymptotic Sarsa', 'Asymptotic Q-learning', 'Asymptotic Expected Sarsa']
A_COLOURS = COLOURS
A_MARKERS = MARKERS
A_LINESTYLES = ['solid'] * 3

def single_data_point(algorithm, alpha, num_runs, episodes):
    rewards = np.zeros((num_runs, episodes))
    for run in tqdm(range(num_runs), leave=False):
        Q = defaultdict(lambda: 0.0)
        for i in range(episodes):
            Q = algorithm(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=alpha, start_Q=Q, log=False)
            rewards[run, i] = ENVIRONMENT.rewards_sum
    rewards = np.average(rewards, axis=0)
    return np.average(rewards)

def plotter(ax, X, algorithm, num_runs, episodes, label, colour, marker, linestyle):
    rewards = [single_data_point(algorithm, ALPHAS[i], num_runs, episodes) for i in tqdm(range(len(ALPHAS)), leave=False)]
    ax.plot(X, rewards, color=colour, label=label, marker=marker, linestyle=linestyle)

if __name__ == '__main__':
    fig = plt.figure()

    ax = fig.add_subplot()

    # Plot interim performance
    for i in tqdm(range(len(ALGORITHMS)), leave=False):
        plotter(ax, ALPHAS, ALGORITHMS[i], NUM_RUNS, EPISODES, LABELS[i], COLOURS[i], MARKERS[i], LINESTYLES[i])
    # Plot asymptotic performance
    for i in tqdm(range(len(ALGORITHMS))):
        plotter(ax, ALPHAS, ALGORITHMS[i], A_NUM_RUNS, A_EPISODES, A_LABELS[i], A_COLOURS[i], A_MARKERS[i], A_LINESTYLES[i])

    ax.set(ylabel='Sum of rewards per episode', xlabel='alpha')
    ax.set_ylim([-160, 0])
    ax.set_xticks(np.linspace(0.1, 1.0, 10))
    ax.legend()

    plt.show()
