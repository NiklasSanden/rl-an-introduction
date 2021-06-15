import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
NUM_RUNS = 10000
EPISODES = 300
ENVIRONMENT = MaximizationBiasExample(actions_in_B=10)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
GAMMA = 1.0
ALPHA = 0.1

if __name__ == '__main__':
    fig = plt.figure()

    Q_learning_left = np.zeros((NUM_RUNS, EPISODES))
    double_Q_learning_left = np.zeros((NUM_RUNS, EPISODES))
    for run in tqdm(range(NUM_RUNS)):
        Q_Q_learning = defaultdict(lambda: 0.0)
        double_Q_1 = defaultdict(lambda: 0.0)
        double_Q_2 = defaultdict(lambda: 0.0)
        for i in range(EPISODES):
            AGENT.counter = 0
            Q_Q_learning = q_learning(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=ALPHA, start_Q=Q_Q_learning, log=False)
            Q_learning_left[run, i] = 100 * (AGENT.counter != 1) # only one action was taken if you went right
            AGENT.counter = 0
            double_Q_1, double_Q_2 = double_q_learning(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=ALPHA, 
                                                       start_Q_1=double_Q_1, start_Q_2=double_Q_2, log=False)
            double_Q_learning_left[run, i] = 100 * (AGENT.counter != 1)
    Q_learning_left = np.average(Q_learning_left, axis=0)
    double_Q_learning_left = np.average(double_Q_learning_left, axis=0)
    X = np.arange(1, EPISODES + 1)

    ax = fig.add_subplot()
    ax.plot(X, Q_learning_left, color='red', label='Q-learning')
    ax.plot(X, double_Q_learning_left, color='lime', label='Double Q-learning')
    ax.plot(X, np.ones(double_Q_learning_left.size) * 5, color='black', label='Optimal', linestyle='dashed')
    ax.set(ylabel='% left actions from A', xlabel='Episodes')
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 5, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '5%', '25%', '50%', '75%', '100%'])
    ax.legend()

    plt.show()
