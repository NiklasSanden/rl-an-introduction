import numpy as np
import matplotlib.pyplot as plt

import sys, os

from numpy.lib.function_base import average
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.agents import *
from utility.control_algorithms import *
from utility.environments import *
from utility.function_approximators import *
from utility.misc import *

# PARAMETERS
NUM_RUNS = 10
EPISODES = 50
ENVIRONMENT = MountainCar(min_x=-1.2, max_x=0.5, min_v=-0.07, max_v=0.07, speed=0.001, gravity=0.0025, freq=3, 
                          start_x_min=-0.6, start_x_max=-0.4, start_v_min=0, start_v_max=0)
AGENT = NumberOfActionsWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.0))
NUM_TILINGS = 8
Q = TileCoding(num_tilings=NUM_TILINGS, dim_sizes=[ENVIRONMENT.max_x - ENVIRONMENT.min_x, ENVIRONMENT.max_v - ENVIRONMENT.min_v], max_size=4096)
GAMMA = 1
NUM_ALPHAS = 10
ALPHAS = [  np.geomspace(0.55,  1.6,    num=NUM_ALPHAS),
            np.geomspace(0.4,   1.6,    num=NUM_ALPHAS),
            np.geomspace(0.3,   1.5,    num=NUM_ALPHAS),
            np.geomspace(0.2,   0.95,   num=NUM_ALPHAS),
            np.geomspace(0.2,   0.65,   num=NUM_ALPHAS)
] # The algorithms will diverge at different alpha values. There is still a risk of divergence though.
N_VALUES = np.logspace(0, 4, num=5, base=2, dtype=int)
COLOURS = ['red', 'green', 'blue', 'black', 'pink']

if __name__ == '__main__':
    fig = plt.figure()
    
    average_reward = np.zeros((NUM_RUNS, len(N_VALUES), NUM_ALPHAS))
    for run in tqdm(range(NUM_RUNS)):
        for n in tqdm(range(len(N_VALUES)), leave=False):
            for a in tqdm(range(NUM_ALPHAS), leave=False):
                Q.zero_weights()
                AGENT.reset_counter()
                Q = semi_gradient_n_step_sarsa(ENVIRONMENT, AGENT, GAMMA, n=N_VALUES[n], max_episodes=EPISODES, alpha=ALPHAS[n][a], Q=Q, log=False)
                average_reward[run, n, a] = AGENT.get_number_of_calls() / EPISODES

    average_reward = np.average(average_reward, axis=0)

    ax = fig.add_subplot()
    for n in range(len(N_VALUES)):
        ax.plot(ALPHAS[n], average_reward[n, :], color=COLOURS[n], label=f'n={N_VALUES[n]}')
    ax.set(ylabel=f'Steps per episode (averaged over first {EPISODES} episodes and {NUM_RUNS} runs)', xlabel=f'alpha * number of tilings ({NUM_TILINGS})')
    ax.set_ylim(220, 300)
    ax.legend()

    plt.show()
