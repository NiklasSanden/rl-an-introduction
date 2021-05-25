import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.action_selectors import *
from utility.agent import *
from utility.environments import *
from utility.q_approximators import *

# PARAMETERS
K = 10
VARIANCE = 1.0
NUM_RUNS = 2000
STEPS = 1000
PARAMETERS = [
    [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
    [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    [1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    [1/4, 1/2, 1, 2, 4]
]
AGENTS = [
    [
        Agent(K, EpsilonGreedyActionSelector(i), QSampleAvg(K)) for i in PARAMETERS[0]
    ],
    [
        GradientBandit(K, i) for i in PARAMETERS[1]
    ],
    [
        Agent(K, UCBActionSelector(K, i), QSampleAvg(K)) for i in PARAMETERS[2]
    ],
    [
        Agent(K, GreedyActionSelector(), QConstantOptimistic(K, 0.1, i)) for i in PARAMETERS[3]
    ]
]
COLOURS = ['r', 'g', 'b', 'k']
LABELS = ['Epsilon', 'Gradient', 'UCB', 'Optimistic']

if __name__ == "__main__":
    agents = AGENTS
    bandits = StationaryKArmedBandits(K, VARIANCE)
    
    average_rewards = [np.zeros((NUM_RUNS, len(agents[i]))) for i in range(len(agents))]
    for n in tqdm(range(NUM_RUNS)):
        for i in range(len(agents)):
            for agent in agents[i]:
                agent.reset()
        bandits.reset()

        rewards = [np.zeros((STEPS, len(agents[i]))) for i in range(len(agents))]
        for j in tqdm(range(STEPS), leave=False):
            for i in range(len(agents)):
                for a in range(len(agents[i])):
                    action = agents[i][a].select_action()
                    reward = bandits.draw(action)
                    agents[i][a].update_action(action, reward)

                    rewards[i][j, a] = reward
        
        for i in range(len(rewards)):
            rewards[i] = np.average(rewards[i], axis=0)
            average_rewards[i][n, :] = rewards[i]

    for i in range(len(rewards)):
        average_rewards[i] = np.average(average_rewards[i], axis=0)

    plt.plot()
    for i in range(len(agents)):
        plt.plot(PARAMETERS[i], average_rewards[i], color=COLOURS[i], label=LABELS[i])
    plt.ylabel('Average reward')
    plt.xlabel('Parameter value')
    plt.legend()
    plt.xscale('log', base=2)
    plt.show()
