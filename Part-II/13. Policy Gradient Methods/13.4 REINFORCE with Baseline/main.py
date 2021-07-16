import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.control_algorithms import *
from utility.environments import *

class ShortCorridorPolicy(object):
    def __init__(self, env):
        self.call_counter = 0
        self.env = env
        self.actions = env.get_actions(None)
        assert len(self.actions) == 2, 'The short corridor environment is only supposed to have 2 actions'
        self.zero_weights()

    def __call__(self, state):
        self.call_counter += 1
        return np.random.choice(self.actions, p=self._soft_max(state))

    def get_ln_gradients(self, state, action):
        soft_max = self._soft_max(state)
        feature_vectors = np.array([self._get_feature_vector(state, a) for a in self.actions])
        return self._get_feature_vector(state, action) - np.sum(soft_max * feature_vectors, axis=0)

    def update_weights(self, alpha, change):
        self.weights = self.weights + alpha * change

    def zero_weights(self):
        # Setting the weights to 0 means it will select right 50%, which is close to optimal (59%)
        # These values were taken from https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter13/short_corridor.py
        self.weights = np.array([1.47, -1.47], dtype=float)

    def get_number_of_calls(self):
        return self.call_counter

    def reset_call_counter(self):
        self.call_counter = 0

    def _soft_max(self, state):
        feature_vectors = np.array([self._get_feature_vector(state, a) for a in self.actions])
        h_values = feature_vectors.dot(self.weights)
        exp_values = np.exp(h_values - max(h_values))
        probs = exp_values / np.sum(exp_values)
        # The episode won't finish if we become deterministic
        if np.min(probs) < 0.01:
            assert len(probs) == 2, 'This part expects there to only be 2 actions'
            best_index = np.argmax(probs)
            probs[:] = 0.01
            probs[best_index] = 0.99
        return probs

    def _get_feature_vector(self, state, action):
        return np.array([1, 0], dtype=float) if action == self.actions[0] else np.array([0, 1], dtype=float)

class ShortCorridorV(object):
    def __init__(self):
        self.zero_weights()

    def __call__(self, state):
        return self.weights[0]

    def get_gradients(self, state):
        return np.ones_like(self.weights, dtype=float)

    def update_weights(self, alpha, change):
        self.weights = self.weights + alpha * change

    def zero_weights(self):
        self.weights = np.zeros(1)

# PARAMETERS
NUM_RUNS = 100
EPISODES = 1000
ENVIRONMENT = ShortCorridor()
AGENT = ShortCorridorPolicy(ENVIRONMENT)
V = ShortCorridorV()
GAMMA = 1
ALPHAS = [2 ** -12, 2 ** -13, 2 ** -14, (2 ** -9, 2 ** -6)]
COLOURS = ['blue', 'red', 'green', 'magenta']
LABELS = ['REINFORCE alpha=2^-12', 'REINFORCE alpha=2^-13', 'REINFORCE alpha=2^-14', 'REINFORCE with baseline alpha_theta=2^-9, alpha_w=2^-6']

if __name__ == '__main__':
    fig = plt.figure()

    average_reward = np.zeros((NUM_RUNS, EPISODES, len(LABELS)))
    for run in tqdm(range(NUM_RUNS), disable=False):
        for a in tqdm(range(len(ALPHAS)), disable=False, leave=False):
            AGENT.zero_weights()
            V.zero_weights()
            for e in range(EPISODES):
                # if no baseline
                if a < len(ALPHAS) - 1:
                    REINFORCE(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, alpha=ALPHAS[a], log=False)
                else:
                    REINFORCE_with_baseline(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, alpha_theta=ALPHAS[a][0], alpha_w=ALPHAS[a][1], V=V, log=False)
                average_reward[run, e, a] = -AGENT.get_number_of_calls()
                AGENT.reset_call_counter()

    average_reward = np.average(average_reward, axis=0)
    
    ax = fig.add_subplot()
    for a in range(len(ALPHAS)):
        ax.plot(np.arange(1, EPISODES + 1), average_reward[:, a], color=COLOURS[a], label=LABELS[a])
    ax.plot(np.arange(1, EPISODES + 1), np.ones(EPISODES) * -11.6, color='gray', linestyle='dashed', label='v*(s0)')
    ax.set(ylabel=f'Total reward on episode (averaged over {NUM_RUNS} runs)', xlabel='Episode')
    ax.set_ylim([-90, -10])
    ax.legend()

    plt.show()