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
MEAN = 4.0
NUM_RUNS = 2000
STEPS = 1000
AGENTS = [
    GradientBandit(K, 0.1),
    Agent(K, UCBActionSelector(K, 2.0), QSampleAvg(K)),
    Agent(K, EpsilonGreedyActionSelector(0.1), QConstant(K, 0.1))
]
COLOURS = ['r', 'b', 'g']
LABELS = ['Gradient', 'UCB', 'Constant Epsilon Greedy']

if __name__ == "__main__":
    agents = AGENTS
    bandits = StationaryMovedKArmedBandits(K, VARIANCE, MEAN)
    average_reward = np.zeros((NUM_RUNS, STEPS, len(agents)))
    best_action_precentage = np.zeros((NUM_RUNS, STEPS, len(agents)))
    for i in tqdm(range(NUM_RUNS)):
        for agent in agents:
            agent.reset()
        bandits.reset()
        best_action = bandits.best_action()

        for j in range(STEPS):
            for a in range(len(agents)):
                action = agents[a].select_action()
                reward = bandits.draw(action)
                agents[a].update_action(action, reward)

                average_reward[i, j, a] = reward
                best_action_precentage[i, j, a] = 100 if action == best_action else 0
    
    average_reward = np.average(average_reward, axis=0)
    best_action_precentage = np.average(best_action_precentage, axis=0)

    fig, axs = plt.subplots(2)
    for i in range(len(agents)):
        axs[0].plot(average_reward[:, i], color=COLOURS[i], label=LABELS[i])
        axs[1].plot(best_action_precentage[:, i], color=COLOURS[i], label=LABELS[i])
    axs[0].set(ylabel='Average reward', xlabel='Steps')
    axs[1].set(ylabel='Optimal action %', xlabel='Steps')
    axs[0].legend()
    axs[1].legend()
    plt.show()
