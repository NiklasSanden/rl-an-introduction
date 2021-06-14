import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

def get_path(Q):
    states = [ENVIRONMENT.reset()]
    terminal = False
    while not terminal:
        next_state, _, terminal, _ = ENVIRONMENT.step(AGENT(states[-1], Q))
        states.append(next_state)
    return states

# PARAMETERS
NUM_RUNS = 1000
EPISODES = 500
ENVIRONMENT = CliffWalkingSumOfRewardsWrapper(rows=4, cols=12)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
GAMMA = 1.0
ALPHA = 0.5
SARSA_COLOUR = 'blue'
Q_COLOUR = 'red'

if __name__ == '__main__':
    fig = plt.figure()

    # Plot sum of rewards per episode during training
    sarsa_rewards = np.zeros((NUM_RUNS, EPISODES))
    q_learning_rewards = np.zeros((NUM_RUNS, EPISODES))
    for run in tqdm(range(NUM_RUNS)):
        Q_sarsa = defaultdict(lambda: 0.0)
        Q_q_learning = defaultdict(lambda: 0.0)
        for i in range(EPISODES):
            Q_sarsa = sarsa_on_policy_td_q(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=ALPHA, start_Q=Q_sarsa, log=False)
            sarsa_rewards[run, i] = ENVIRONMENT.rewards_sum
            Q_q_learning = q_learning(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=ALPHA, start_Q=Q_q_learning, log=False)
            q_learning_rewards[run, i] = ENVIRONMENT.rewards_sum
    sarsa_rewards = np.average(sarsa_rewards, axis=0)
    q_learning_rewards = np.average(q_learning_rewards, axis=0)    
    X = np.arange(1, EPISODES + 1)

    ax = fig.add_subplot()
    ax.plot(X, sarsa_rewards, color=SARSA_COLOUR, label='Sarsa')
    ax.plot(X, q_learning_rewards, color=Q_COLOUR, label='Q-learning')
    ax.set(ylabel='Sum of rewards during episode', xlabel='Episodes')
    ax.set_ylim([-100, -20])
    ax.legend()

    # Get preferred paths
    AGENT.epsilon = 0.0
    print('Sarsas deterministic path is:', get_path(Q_sarsa))
    print('Q-learnings deterministic path is:', get_path(Q_q_learning))

    plt.show()
