import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *

# PARAMETERS
EPISODES = 10000
ENVIRONMENT = ShortCorridor()
PROBABILITIES_RIGHT = np.linspace(0.05, 0.95, num=50)

if __name__ == '__main__':
    fig = plt.figure()

    average_reward = np.zeros((EPISODES, len(PROBABILITIES_RIGHT)))
    for prob in tqdm(range(len(PROBABILITIES_RIGHT))):
        for e in tqdm(range(EPISODES), leave=False):
            steps = 0
            start_state = ENVIRONMENT.reset()
            terminal = False
            while not terminal:
                _, _, terminal, _ = ENVIRONMENT.step(np.random.choice(ENVIRONMENT.get_actions(start_state), p=[1 - PROBABILITIES_RIGHT[prob], PROBABILITIES_RIGHT[prob]]))
                steps += 1
            average_reward[e, prob] = -steps
    
    average_reward = np.average(average_reward, axis=0)
    
    ax = fig.add_subplot()
    ax.plot(PROBABILITIES_RIGHT, average_reward, color='black')
    ax.plot([PROBABILITIES_RIGHT[0]], [average_reward[0]], color='red', marker='o', label='Epsilon-greedy left')
    ax.plot([PROBABILITIES_RIGHT[-1]], [average_reward[-1]], color='blue', marker='o', label='Epsilon-greedy right')
    
    best_index = np.argmax(average_reward)
    ax.plot([PROBABILITIES_RIGHT[best_index]], [average_reward[best_index]], color='green', marker='o', 
            label=f'Best stochastic policy found, p={round(PROBABILITIES_RIGHT[best_index], 2)}')

    ax.set(ylabel=f'Reward per episode (averaged over {EPISODES} episodes)', xlabel='probability of right action')
    ax.legend()

    plt.show()