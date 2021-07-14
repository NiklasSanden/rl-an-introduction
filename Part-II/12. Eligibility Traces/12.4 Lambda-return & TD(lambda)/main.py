import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.agents import *
from utility.environments import *
from utility.function_approximators import *
from utility.misc import *
from utility.policy_evaluation import *

# PARAMETERS
NUM_RUNS = 20
EPISODES = 10
NUM_STATES = 19
ENVIRONMENT = RandomWalk(num_states=NUM_STATES)
AGENT = RepeatableWrapper(EpsilonGreedyAgent(ENVIRONMENT, epsilon=1.0))
V = Tabular(num_states=NUM_STATES)
GAMMA = 1
ALGORITHMS = [offline_lambda_return_v, online_lambda_return_v, TD_lambda_v]
TRUE_VALUES = [i / (NUM_STATES + 1) for i in range(-NUM_STATES + 1, NUM_STATES + 1, 2)]
ALPHAS = [x - 0.4 for x in np.geomspace(0.4, 1.4, num=25)]
LAMBDAS = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
COLOURS = ['red', 'green', 'blue', 'black', 'pink', 'lightblue', 'purple', 'magenta']
TITLES = ['Offline lambda-return', 'Online lambda-return', 'TD(lambda)']

if __name__ == '__main__':
    fig = plt.figure()

    errors = np.zeros((NUM_RUNS, EPISODES, len(ALPHAS), len(LAMBDAS), len(ALGORITHMS)))
    for run in tqdm(range(NUM_RUNS)):
        AGENT.reset_actions()
        for a in tqdm(range(len(ALPHAS)), leave=False):
            for algo in range(len(ALGORITHMS)):
                for l in range(len(LAMBDAS)):
                    AGENT.repeat()
                    V.zero_weights()
                    for episode in range(EPISODES):
                        V = ALGORITHMS[algo](ENVIRONMENT, AGENT, GAMMA, lambda_=LAMBDAS[l], max_episodes=1, alpha=ALPHAS[a], V=V, log=False)
                        errors[run, episode, a, l, algo] = RMS_error(V, [x for x in range(NUM_STATES)], TRUE_VALUES)
    
    errors = np.average(errors, axis=(0, 1))
    
    for algo in range(len(ALGORITHMS)):
        ax = fig.add_subplot(1, len(ALGORITHMS), algo + 1)
        for l in range(len(LAMBDAS)):
            ax.plot(ALPHAS, errors[:, l, algo], color=COLOURS[l], label=f'lambda={LAMBDAS[l]}')
        if algo == 0:
            ax.set(ylabel=f'Average RMS error of the first {EPISODES} episodes')
        ax.set(title=TITLES[algo], xlabel='alpha')
        ax.set_ylim([0.25, 0.55])
        ax.legend()

    plt.show()