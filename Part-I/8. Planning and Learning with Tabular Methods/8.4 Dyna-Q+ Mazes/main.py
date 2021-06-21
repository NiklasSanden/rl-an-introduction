import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
NUM_RUNS = 20
TIME_STEPS = 3000
ROWS = 6
COLS = 9
ENVIRONMENT = CumulativeRewardPerTimestepWrapper(ChangingMaze(rows=ROWS, cols=COLS, start=(5, 3), end=(0, 8), blocks=[(3, c) for c in range(8)],
                                                              changed_blocks=[(3, c) for c in range(1, 9)], time_steps_until_change=1000))
DYNA_Q_AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
DYNA_Q_PLUS_AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.0)
N = 50
KAPPA = 0.001
GAMMA = 0.95
ALPHA = 1.0
COLOURS = ['red', 'green', 'blue']
LABELS = ['Dyna-Q+', 'Dyna-Q+ alternative', 'Dyna-Q']

def call_algorithms(algo, run, cumulative_reward):
    ENVIRONMENT.environment.reset_change()
    ENVIRONMENT.reset_rewards()

    Q = defaultdict(lambda: 0)
    time_step = 0
    if algo == 2:
        model = dict()
        model_keys = []
    else:
        model = {((r, c), a): (0, (r, c)) for r in range(ROWS) for c in range(COLS) for a in ENVIRONMENT.get_actions((r, c))}
        model_keys = list(model)
        tau = {((r, c), a): 0 for r in range(ROWS) for c in range(COLS) for a in ENVIRONMENT.get_actions((r, c))}
    while time_step < TIME_STEPS:
        if algo == 0:
            Q, model, model_keys, tau, time_step = dyna_Q_plus(ENVIRONMENT, DYNA_Q_PLUS_AGENT, GAMMA, max_episodes=1, n=N, kappa=KAPPA, alpha=ALPHA, Q=Q, 
                                                               model=model, model_keys=model_keys, tau=tau, time_step=time_step, log=False)
        elif algo == 1:
            Q, model, model_keys, tau, time_step = dyna_Q_plus_greedy_exercise_8_4(ENVIRONMENT, GAMMA, max_episodes=1, n=N, kappa=KAPPA, alpha=ALPHA, Q=Q, 
                                                                                   model=model, model_keys=model_keys, tau=tau, time_step=time_step, log=False)
        else:
            Q, model, model_keys = dyna_Q(ENVIRONMENT, DYNA_Q_AGENT, GAMMA, max_episodes=1, n=N, alpha=ALPHA, Q=Q, model=model, model_keys=model_keys, log=False)
            time_step = len(ENVIRONMENT.get_cumulative_rewards())
    rewards = ENVIRONMENT.get_cumulative_rewards()
    cumulative_reward[run, :, algo] = rewards[:TIME_STEPS]

if __name__ == '__main__':
    fig = plt.figure()

    cumulative_reward = np.zeros((NUM_RUNS, TIME_STEPS, len(LABELS)))
    for run in tqdm(range(NUM_RUNS)):
        for a in range(len(LABELS)):
            call_algorithms(a, run, cumulative_reward)
    
    cumulative_reward = np.average(cumulative_reward, axis=0)

    ax = fig.add_subplot()
    for a in range(len(LABELS)):
        ax.plot(cumulative_reward[:, a], color=COLOURS[a], label=LABELS[a])
    ax.set(ylabel='Cumulative reward', xlabel='Time steps')
    ax.legend()

    plt.show()
