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
EPISODES = 20
ENVIRONMENT = MaxStepsWrapper(MountainCar(min_x=-1.2, max_x=0.5, min_v=-0.07, max_v=0.07, speed=0.001, gravity=0.0025, freq=3, 
                                          start_x_min=-0.6, start_x_max=-0.4, start_v_min=0, start_v_max=0), 
                              max_steps=5000)
AGENT = NumberOfActionsWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.0))
NUM_TILINGS = 8
Q = TileCoding(num_tilings=NUM_TILINGS, dim_sizes=[ENVIRONMENT.env.max_x - ENVIRONMENT.env.min_x, ENVIRONMENT.env.max_v - ENVIRONMENT.env.min_v], max_size=4096)
GAMMA = 1
NUM_ALPHAS = 10
ALPHAS = [  np.linspace(0.2, 2, num=NUM_ALPHAS),
            [0.2, 0.4, 0.6],
            np.linspace(0.2, 2, num=NUM_ALPHAS)
] # Accumulating traces diverges with high alphas, so there is no point in plotting it.
LAMBDA = 0.9
COLOURS = ['green', 'black', 'red']
LABELS = ['True online Sarsa(lambda)', 'Sarsa(lambda) with accumulating traces', 'Sarsa(lambda) with replacing traces']
MARKERS = ['s', 'o', 'v']

if __name__ == '__main__':
    fig = plt.figure()

    assert NUM_ALPHAS >= len(ALPHAS[1]), f'Accumulating traces has more than {NUM_ALPHAS} alpha values'
    
    # Some values won't be used since accumulating traces has a lower number of alpha values.
    average_reward = np.zeros((NUM_RUNS, NUM_ALPHAS, len(COLOURS)))
    for run in tqdm(range(NUM_RUNS)):
        for a in tqdm(range(NUM_ALPHAS), leave=False):
            # True online Sarsa(lambda)
            Q.zero_weights()
            AGENT.reset_counter()
            Q = true_online_sarsa_lambda(ENVIRONMENT, AGENT, GAMMA, lambda_=LAMBDA, max_episodes=EPISODES, alpha=ALPHAS[0][a] / NUM_TILINGS, Q=Q, log=False)
            average_reward[run, a, 0] = -AGENT.get_number_of_calls() / EPISODES

            # Sarsa(lambda) with accumulating traces
            if a < len(ALPHAS[1]):
                Q.zero_weights()
                AGENT.reset_counter()
                Q = semi_gradient_sarsa_lambda(ENVIRONMENT, AGENT, GAMMA, lambda_=LAMBDA, max_episodes=EPISODES, alpha=ALPHAS[1][a] / NUM_TILINGS, 
                                               Q=Q, replacing_traces=False, log=False)
                average_reward[run, a, 1] = -AGENT.get_number_of_calls() / EPISODES
            
            # Sarsa(lambda) with replacing traces
            Q.zero_weights()
            AGENT.reset_counter()
            Q = semi_gradient_sarsa_lambda(ENVIRONMENT, AGENT, GAMMA, lambda_=LAMBDA, max_episodes=EPISODES, alpha=ALPHAS[2][a] / NUM_TILINGS, 
                                           Q=Q, replacing_traces=True, log=False)
            average_reward[run, a, 2] = -AGENT.get_number_of_calls() / EPISODES

    average_reward = np.average(average_reward, axis=0)

    ax = fig.add_subplot()
    for a in range(len(COLOURS)):
        ax.plot(ALPHAS[a], average_reward[:len(ALPHAS[a]), a], color=COLOURS[a], label=LABELS[a], marker=MARKERS[a])
    ax.set(ylabel=f'Reward per episode (averaged over first {EPISODES} episodes and {NUM_RUNS} runs)', 
           xlabel=f'alpha * number of tilings ({NUM_TILINGS})')
    ax.set_ylim(-550, -150)
    ax.set_xticks(ALPHAS[0])
    ax.legend()

    plt.show()
