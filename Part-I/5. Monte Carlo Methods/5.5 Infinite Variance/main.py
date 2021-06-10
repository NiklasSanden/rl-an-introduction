import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_iteration import *

class TargetPolicy(object):
    def __call__(self, state):
        return 0
    
    def get_prob(self, action, state):
        a = self.__call__(state)
        return 1.0 if a == action else 0.0

class BehaviourPolicy(object):
    def __call__(self, state):
        return 1 if np.random.rand() < 0.5 else 0

    def get_prob(self, action, state):
        return 0.5

# PARAMETERS
NUM_RUNS = 10
EPISODES = 10000000
ENVIRONMENT = InfiniteVarianceLoop()
TARGET = TargetPolicy()
BEHAVIOUR = BehaviourPolicy()
GAMMA = 1.0

if __name__ == '__main__':
    fig = plt.figure()

    v_estimates = np.zeros((NUM_RUNS, EPISODES))
    for i in tqdm(range(NUM_RUNS), disable=False):
        V = defaultdict(lambda: 0.0)
        C = defaultdict(lambda: 0.0)
        for j in tqdm(range(EPISODES), leave=False):
            V, C = monte_carlo_v_off_policy(ENVIRONMENT, TARGET, BEHAVIOUR, GAMMA, max_iterations=1, start_V=V, start_C=C, weighted=False)
            v_estimates[i, j] = V[0]

    ax = fig.add_subplot()
    for i in range(NUM_RUNS):
        ax.plot(v_estimates[i, :])
    ax.set(ylabel='Estimate of V for ' + str(NUM_RUNS) + ' different runs', xlabel='Episodes (log scale)')
    ax.set_xscale('log', base=10)

    plt.show()
