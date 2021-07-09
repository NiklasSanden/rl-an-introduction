import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.agents import *
from utility.control_algorithms import *
from utility.environments import *
from utility.function_approximators import *
from utility.misc import *

# PARAMETERS
NUM_RUNS = 100
EPISODES = 500
ENVIRONMENT = MountainCar(min_x=-1.2, max_x=0.5, min_v=-0.07, max_v=0.07, speed=0.001, gravity=0.0025, freq=3, 
                          start_x_min=-0.6, start_x_max=-0.4, start_v_min=0, start_v_max=0)
AGENT = NumberOfActionsWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1))
NUM_TILINGS = 8
Q = TileCoding(num_tilings=NUM_TILINGS, dim_sizes=[ENVIRONMENT.max_x - ENVIRONMENT.min_x, ENVIRONMENT.max_v - ENVIRONMENT.min_v], max_size=4096)
GAMMA = 1
ALPHAS = [0.1, 0.2, 0.5]
COLOURS = ['blue', 'green', 'red']

if __name__ == '__main__':
    fig = plt.figure()
    
    steps_per_episode = np.zeros((NUM_RUNS, EPISODES, len(ALPHAS)))
    for run in tqdm(range(NUM_RUNS)):
        for a in tqdm(range(len(ALPHAS)), leave=False):
            Q.zero_weights()
            for e in tqdm(range(EPISODES), leave=False):
                AGENT.reset_counter()
                Q = semi_gradient_sarsa(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, alpha=ALPHAS[a], Q=Q, log=False)
                steps_per_episode[run, e, a] = AGENT.get_number_of_calls()

    steps_per_episode = np.average(steps_per_episode, axis=0)

    ax = fig.add_subplot()
    for a in range(len(ALPHAS)):
        ax.plot(np.arange(1, EPISODES + 1), steps_per_episode[:, a], color=COLOURS[a], label='alpha=' + str(ALPHAS[a]) + '/' + str(NUM_TILINGS))
    ax.set(ylabel='Steps per episode log scale (averaged over 100 runs)', xlabel='Episodes')
    ax.set_yscale('log', base=10)
    Y = [100, 200, 400, 1000]
    ax.set_ylim(Y[0], Y[-1])
    ax.set_yticks(Y)
    ax.set_yticklabels(Y)
    ax.set_yticks([], minor=True)
    ax.legend()

    plt.show()
