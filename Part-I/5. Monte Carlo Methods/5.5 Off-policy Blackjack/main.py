import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_iteration import *

class TargetPolicy(object):
    def __init__(self, stick_after=20):
        self.stick_after = stick_after

    def __call__(self, state):
        _, player_sum, _ = state
        return 0 if player_sum < self.stick_after else 1
    
    def get_prob(self, action, state):
        a = self.__call__(state)
        return 1.0 if a == action else 0.0

class BehaviourPolicy(object):
    def __call__(self, state):
        return 1 if np.random.rand() < 0.5 else 0

    def get_prob(self, action, state):
        return 0.5

# PARAMETERS
NUM_RUNS = 100
EPISODES = 10000
(DEALER_START, PLAYER_START, USABLE_ACE_START) = (2, 13, True) # state to evaluate
ENVIRONMENT = BlackjackSingleStartState(DEALER_START, PLAYER_START, USABLE_ACE_START)
TARGET = TargetPolicy(stick_after=20)
BEHAVIOUR = BehaviourPolicy()
COLOURS = ['g', 'r']
LABELS = ['Ordinary', 'Weighted']
GAMMA = 1.0
TRUE_VALUE = -0.27726 # See page 106 in the textbook

if __name__ == '__main__':
    fig = plt.figure()

    average_error = np.zeros((NUM_RUNS, EPISODES, len(LABELS)))
    for i in tqdm(range(NUM_RUNS), disable=False):
        ordinary_V = defaultdict(lambda: 0.0)
        ordinary_C = defaultdict(lambda: 0.0)
        weighted_V = defaultdict(lambda: 0.0)
        weighted_C = defaultdict(lambda: 0.0)
        for j in range(EPISODES):
            ordinary_V, ordinary_C = monte_carlo_v_off_policy(ENVIRONMENT, TARGET, BEHAVIOUR, GAMMA, max_iterations=1, start_V=ordinary_V, start_C=ordinary_C, weighted=False)
            weighted_V, weighted_C = monte_carlo_v_off_policy(ENVIRONMENT, TARGET, BEHAVIOUR, GAMMA, max_iterations=1, start_V=weighted_V, start_C=weighted_C, weighted=True)
            average_error[i, j, 0] = (ordinary_V[(DEALER_START, PLAYER_START, USABLE_ACE_START)] - TRUE_VALUE) ** 2
            average_error[i, j, 1] = (weighted_V[(DEALER_START, PLAYER_START, USABLE_ACE_START)] - TRUE_VALUE) ** 2

    average_error = np.average(average_error, axis=0)

    ax = fig.add_subplot()
    for i in range(len(LABELS)):
        ax.plot(average_error[:, i], color=COLOURS[i], label=LABELS[i])
    ax.set(ylabel='Mean square error (average over 100 runs)', xlabel='Episodes (log scale)')
    ax.legend()
    ax.set_xscale('log', base=10)
    ax.set_ylim([0, 5])

    plt.show()
