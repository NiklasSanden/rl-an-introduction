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
EPISODES = 20000
AVERAGE_X_EPISODES = 1000
ENVIRONMENT = WindyGridworld(rows=7, cols=10, wind_values=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0], is_stochastic=True, king_moves=True, can_pass=False)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
GAMMA = 1.0
ALPHA = 0.05

if __name__ == '__main__':
    fig = plt.figure()

    # Get plot of improvement
    cumulative_time_steps = np.zeros(EPISODES)
    Q = defaultdict(lambda: 0.0)
    for i in tqdm(range(EPISODES)):
        Q = sarsa_on_policy_td_q(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=ALPHA, start_Q=Q, log=False)
        cumulative_time_steps[i] = AGENT.counter
    episodes_completed_at_time_step = np.zeros(round(cumulative_time_steps[-1]) + 1)
    index = 0
    for i in range(episodes_completed_at_time_step.size):
        if i <= cumulative_time_steps[index]:
            episodes_completed_at_time_step[i] = index
        else:
            index += 1
            episodes_completed_at_time_step[i] = index

    print('The last', AVERAGE_X_EPISODES, 'episodes had an average of', np.average(cumulative_time_steps[-AVERAGE_X_EPISODES:] - cumulative_time_steps[-AVERAGE_X_EPISODES-1:-1]), 'time steps')

    # Get greedy path (take average since the environment might be stochastic)
    AGENT.epsilon = 0.0
    AGENT.counter = 0
    greedy_cumulative_time_steps = np.zeros(1000)
    for i in range(1, greedy_cumulative_time_steps.size):
        Q = sarsa_on_policy_td_q(ENVIRONMENT, AGENT, GAMMA, max_iterations=1, alpha=ALPHA, start_Q=Q, log=False)
        greedy_cumulative_time_steps[i] = AGENT.counter
    print('Optimal path yields', np.average(greedy_cumulative_time_steps[1:] - greedy_cumulative_time_steps[:-1]), 'time steps')

    ax = fig.add_subplot()
    ax.plot(episodes_completed_at_time_step, color='red')
    ax.set(ylabel='Episodes', xlabel='Time steps')

    plt.show()
