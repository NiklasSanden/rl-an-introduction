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
AGENT = NumberOfActionsWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.0))
NUM_TILINGS = 8
Q = TileCoding(num_tilings=NUM_TILINGS, dim_sizes=[ENVIRONMENT.max_x - ENVIRONMENT.min_x, ENVIRONMENT.max_v - ENVIRONMENT.min_v], max_size=4096)
GAMMA = 1
ALPHAS = [0.5, 0.3]
N_VALUES = [1, 8]
COLOURS = ['red', 'purple']

if __name__ == '__main__':
    fig = plt.figure()
    
    steps_per_episode = np.zeros((NUM_RUNS, EPISODES, len(N_VALUES)))
    for run in tqdm(range(NUM_RUNS)):
        for n in tqdm(range(len(N_VALUES)), leave=False):
            Q.zero_weights()
            for e in tqdm(range(EPISODES), leave=False):
                AGENT.reset_counter()
                Q = semi_gradient_n_step_sarsa(ENVIRONMENT, AGENT, GAMMA, n=N_VALUES[n], max_episodes=1, alpha=ALPHAS[n], Q=Q, log=False)
                steps_per_episode[run, e, n] = AGENT.get_number_of_calls()

    steps_per_episode = np.average(steps_per_episode, axis=0)

    ax = fig.add_subplot()
    for n in range(len(N_VALUES)):
        ax.plot(np.arange(1, EPISODES + 1), steps_per_episode[:, n], color=COLOURS[n], label=f'n={N_VALUES[n]}')
    ax.set(ylabel=f'Steps per episode log scale (averaged over {NUM_RUNS} runs)', xlabel='Episodes')
    ax.set_yscale('log', base=10)
    Y = [100, 200, 400, 1000]
    ax.set_ylim(Y[0], Y[-1])
    ax.set_yticks(Y)
    ax.set_yticklabels(Y)
    ax.set_yticks([], minor=True)
    ax.legend()

    plt.show()
