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
NUM_RUNS = 10
EPISODES = 50
ENVIRONMENT = MountainCar(min_x=-1.2, max_x=0.5, min_v=-0.07, max_v=0.07, speed=0.001, gravity=0.0025, freq=3, 
                          start_x_min=-0.6, start_x_max=-0.4, start_v_min=0, start_v_max=0)
AGENT = NumberOfActionsWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.0))
NUM_TILINGS = 8
Q = TileCoding(num_tilings=NUM_TILINGS, dim_sizes=[ENVIRONMENT.max_x - ENVIRONMENT.min_x, ENVIRONMENT.max_v - ENVIRONMENT.min_v], max_size=4096)
REPLACING_TRACES = True
GAMMA = 1
NUM_ALPHAS = 10
ALPHAS = [  np.geomspace(0.40,  1.75,    num=NUM_ALPHAS),
            np.geomspace(0.30,  1.75,    num=NUM_ALPHAS),
            np.geomspace(0.25,  1.75,    num=NUM_ALPHAS),
            np.geomspace(0.25,  1.75,    num=NUM_ALPHAS),
            np.geomspace(0.25,  1.75,    num=NUM_ALPHAS),
            np.geomspace(0.25,  1.70,    num=NUM_ALPHAS),
            np.geomspace(0.25,  1.50,    num=NUM_ALPHAS)
] # The algorithms will diverge at different alpha values, so you can change them here. 
  # These values seem fine with replacing traces, but will need to be changed for accumulating traces.
LAMBDAS = [0, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99]
COLOURS = ['purple', 'cyan', 'pink', 'green', 'red', 'blue', 'black']

if __name__ == '__main__':
    fig = plt.figure()
    
    average_reward = np.zeros((NUM_RUNS, len(LAMBDAS), NUM_ALPHAS))
    for run in tqdm(range(NUM_RUNS)):
        for l in tqdm(range(len(LAMBDAS)), leave=False):
            for a in tqdm(range(NUM_ALPHAS), leave=False):
                Q.zero_weights()
                AGENT.reset_counter()
                Q = semi_gradient_sarsa_lambda(ENVIRONMENT, AGENT, GAMMA, lambda_=LAMBDAS[l], max_episodes=EPISODES, alpha=ALPHAS[l][a], 
                                               Q=Q, replacing_traces=REPLACING_TRACES, log=False)
                average_reward[run, l, a] = AGENT.get_number_of_calls() / EPISODES

    average_reward = np.average(average_reward, axis=0)

    ax = fig.add_subplot()
    for l in range(len(LAMBDAS)):
        ax.plot(ALPHAS[l], average_reward[l, :], color=COLOURS[l], label=f'lambda={LAMBDAS[l]}')
    ax.set(title='Sarsa(lambda) with replacing traces', 
           ylabel=f'Steps per episode (averaged over first {EPISODES} episodes and {NUM_RUNS} runs)', 
           xlabel=f'alpha * number of tilings ({NUM_TILINGS})')
    ax.set_ylim(160, 300)
    ax.legend()

    plt.show()
