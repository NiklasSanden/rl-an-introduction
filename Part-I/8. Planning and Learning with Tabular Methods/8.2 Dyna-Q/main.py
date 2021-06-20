import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
NUM_RUNS = 30
EPISODES = 50
ENVIRONMENT = DynaMaze()
AGENT = NumberOfActionsWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1))
N_VALUES = [0, 5, 50]
GAMMA = 0.95
ALPHA = 0.1
COLOURS = ['cyan', 'green', 'red']

if __name__ == '__main__':
    fig = plt.figure()

    agent_performance = np.zeros((NUM_RUNS, EPISODES, len(N_VALUES)))
    for run in tqdm(range(NUM_RUNS)):
        for n in range(len(N_VALUES)):
            Q = defaultdict(lambda: 0)
            model = dict()
            model_keys = []
            for episode in range(EPISODES):
                AGENT.reset()
                Q, model, model_keys = dyna_Q(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, n=N_VALUES[n], alpha=ALPHA, Q=Q, model=model, model_keys=model_keys, log=False)
                agent_performance[run, episode, n] = AGENT.get_number_of_calls()
    
    agent_performance = np.average(agent_performance, axis=0)

    ax = fig.add_subplot()
    for n in range(len(N_VALUES)):
        ax.plot(np.arange(2, agent_performance.shape[0] + 1), agent_performance[1:, n], color=COLOURS[n], label='n=' + str(N_VALUES[n]))
    ax.set(ylabel='Steps per episode', xlabel='Episodes')
    ax.legend()

    plt.show()
