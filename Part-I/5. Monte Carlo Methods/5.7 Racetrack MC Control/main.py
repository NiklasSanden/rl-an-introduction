import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_iteration import *

class EpsilonGreedyAgent(object):
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: 0.0)

    def __call__(self, state):
        actions = self.env.get_actions(state)
        if np.random.rand() < self.epsilon:
            return actions[np.random.choice(np.arange(len(actions)))]
        else:
            best_index = np.argmax(np.array([self.Q[(state, a)] for a in actions]))
            return actions[best_index]

    def update_Q(self, Q):
        self.Q = Q

    def get_prob(self, action, state):
        actions = self.env.get_actions(state)
        greedy_action = actions[np.argmax(np.array([self.Q[(state, a)] for a in actions]))]
        non_greedy_prob = self.epsilon / len(actions)
        return non_greedy_prob if action != greedy_action else (1.0 - self.epsilon) + non_greedy_prob

class GreedyAgent(object):
    def __init__(self, env):
        self.env = env
        self.Q = defaultdict(lambda: 0.0)

    def __call__(self, state):
        actions = self.env.get_actions(state)
        best_index = np.argmax(np.array([self.Q[(state, a)] for a in actions]))
        return actions[best_index]

    def update_Q(self, Q):
        self.Q = Q

# PARAMETERS - behaviour is ignored and target is used as agent when OFF_POLICY = False
EPISODES = 20000
MAX_Y_SPEED, MAX_X_SPEED = (4, 4)
ENVIRONMENT = Racetrack_1(MAX_Y_SPEED, MAX_X_SPEED)
TARGET = GreedyAgent(ENVIRONMENT)
BEHAVIOUR = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
GAMMA = 1.0
NUM_START_STATES = 6
OFF_POLICY = True

if __name__ == '__main__':
    if OFF_POLICY:
        # Having pessimistic Q-values is essential for off-policy and is preferred for on-policy as well
        monte_carlo_q_off_policy_control(ENVIRONMENT, TARGET, BEHAVIOUR, GAMMA, max_iterations=EPISODES, target_Q=defaultdict(lambda: -1000000), behaviour_Q=defaultdict(lambda: -1000000))
    else:
        monte_carlo_q_on_policy_control(ENVIRONMENT, TARGET, GAMMA, max_iterations=EPISODES, start_Q=defaultdict(lambda: -1000000))

    tested_states = set()
    # If using off-policy with a deterministic target policy - this could loop indefinitely if it hasn't trained "long enough".
    # Perhaps add some randomness or check if there is a cycle to try to avoid it.
    while len(tested_states) < NUM_START_STATES: 
        state = ENVIRONMENT.reset()
        if state in tested_states:
            continue
        tested_states.add(state)

        track = 10 * ENVIRONMENT.track
        iterator = 10
        terminal = False
        while not terminal:
            iterator += 1
            track[(state[0], state[1])] = iterator
            action = TARGET(state)
            # step_after_noise avoids the randomness of the environment (check its source)
            state, reward, terminal, _ = ENVIRONMENT.step_after_noise(action)
        
        plt.imshow(track, interpolation='none')
        plt.colorbar()
        plt.show()
