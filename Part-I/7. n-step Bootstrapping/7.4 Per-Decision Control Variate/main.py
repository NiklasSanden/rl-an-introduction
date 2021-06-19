import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *
from utility.agents import *
from utility.control_algorithms import *

# PARAMETERS
NUM_RUNS = 50
EPISODES = 200
ENVIRONMENT = WindyGridworldPositiveRewardWrapper(rows=5, cols=8, wind_values=[0, 0, 1, 1, 2, 2, 1, 0], is_stochastic=False, king_moves=False, can_pass=False)
AGENT = GreedyAgent(ENVIRONMENT)
BEHAVIOUR = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
N_VALUES = [1, 3, 6]
ALGORITHMS = [n_step_sarsa_off_policy, n_step_sarsa_off_policy_control_variate]
GAMMA = 0.95
ALPHAS = [0.5, 0.35, 0.2]
COLOURS = ['red', 'green', 'blue']
LINESTYLES = ['solid', 'dashed']
ALGORITHM_LABELS = ['Importance Sampling', 'Control Variate']

def get_agent_performance(Q):
    state = ENVIRONMENT.reset()
    upper_bound = 100 # upper bound in case it doesn't reach the goal
    for i in range(1, upper_bound + 1):
        state, _, terminal, _ = ENVIRONMENT.step(AGENT(state, Q))
        if terminal:
            break
    return i

if __name__ == '__main__':
    fig = plt.figure()

    agent_performance = np.zeros((NUM_RUNS, EPISODES, len(N_VALUES), len(ALGORITHMS)))
    for run in tqdm(range(NUM_RUNS)):
        for n in range(len(N_VALUES)):
            for a in range(len(ALGORITHMS)):
                Q = defaultdict(lambda: 0.0)
                for episode in range(EPISODES):
                    Q = ALGORITHMS[a](ENVIRONMENT, AGENT, BEHAVIOUR, GAMMA, max_iterations=1, n=N_VALUES[n], alpha=ALPHAS[n], start_Q=Q, log=False)
                    agent_performance[run, episode, n, a] = get_agent_performance(Q)
    
    agent_performance = np.average(agent_performance, axis=0)
    
    ax = fig.add_subplot()
    for a in range(len(ALGORITHMS)):
        for n in range(len(N_VALUES)):
            ax.plot(agent_performance[:, n, a], linestyle=LINESTYLES[a], color=COLOURS[n], label=ALGORITHM_LABELS[a] + ' n=' + str(N_VALUES[n]))
    ax.set(ylabel='Episode length when following the target policy (max 100)', xlabel='Episodes of experience')
    ax.set_yscale('log', base=10)
    ax.legend()

    plt.show()
