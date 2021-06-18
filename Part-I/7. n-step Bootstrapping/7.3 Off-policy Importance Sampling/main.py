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
NUM_RUNS = 10
EPISODES = 200
ENVIRONMENT = WindyGridworld(rows=7, cols=10, wind_values=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0], is_stochastic=False, king_moves=False, can_pass=False)
AGENT = GreedyAgent(ENVIRONMENT)
BEHAVIOUR = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
N_VALUES = [1, 2, 4, 8, 16]
GAMMA = 1.0
ALPHAS = [0.5, 0.35, 0.2, 0.1, 0.05]
COLOURS = ['red', 'green', 'blue', 'cyan', 'orange']

if __name__ == '__main__':
    fig = plt.figure()

    episode_length = np.zeros((NUM_RUNS, EPISODES, len(N_VALUES)))
    for run in tqdm(range(NUM_RUNS)):
        for n in range(len(N_VALUES)):
            Q = defaultdict(lambda: 0.0)
            for episode in range(EPISODES):
                BEHAVIOUR.counter = 0
                Q = n_step_sarsa_off_policy(ENVIRONMENT, AGENT, BEHAVIOUR, GAMMA, max_iterations=1, n=N_VALUES[n], alpha=ALPHAS[n], start_Q=Q, log=False)
                episode_length[run, episode, n] = BEHAVIOUR.counter
    
    episode_length = np.average(episode_length, axis=0)
    
    ax = fig.add_subplot()
    for n in range(len(N_VALUES)):
        ax.plot(episode_length[:, n], color=COLOURS[n], label='n=' + str(N_VALUES[n]))
    ax.set(ylabel='Episode length averaged over' + str(NUM_RUNS) + ' runs (log scale)', xlabel='Episode')
    ax.set_yscale('log', base=10)
    ax.legend()

    plt.show()
