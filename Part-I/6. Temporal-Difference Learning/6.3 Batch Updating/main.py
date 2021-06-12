import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_evaluation import *
from utility.misc import *

# PARAMETERS
NUM_RUNS = 100
EPISODES = 100
NUM_STATES = 5
ENVIRONMENT = RandomWalk(num_states=NUM_STATES)
AGENT = lambda s: np.random.choice(ENVIRONMENT.get_actions(s))
MAX_DELTA = 0.0001
GAMMA = 1.0
TRUE_VALUES = [i / (NUM_STATES + 1) for i in range(1, NUM_STATES + 1)]
TD_COLOUR = 'blue'
MC_COLOUR = 'red'

if __name__ == '__main__':
    # Note that the starting estimate of states is 0 while the book might have used 0.5 (it is not stated explicitly in this example)
    fig = plt.figure()

    TD_error = np.zeros((NUM_RUNS, EPISODES))
    MC_error = np.zeros((NUM_RUNS, EPISODES))
    for run in tqdm(range(NUM_RUNS)):
        returns = {}
        experience = {}
        for i in range(EPISODES):
            returns, experience = gather_experience_and_returns(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, start_returns=returns, 
                                                                start_experience_tuples=experience, log=False)
            V = batch_TD0(experience, GAMMA, MAX_DELTA, log=False)
            TD_error[run, i] = RMS_error(V, NUM_STATES, TRUE_VALUES)
            V = batch_monte_carlo(returns)
            MC_error[run, i] = RMS_error(V, NUM_STATES, TRUE_VALUES)
    TD_error = np.average(TD_error, axis=0)
    MC_error = np.average(MC_error, axis=0)
    X = np.arange(1, EPISODES + 1)

    ax = fig.add_subplot()
    ax.plot(X, TD_error, color=TD_COLOUR, label='TD')
    ax.plot(X, MC_error, color=MC_COLOUR, label='MC')
    ax.set(title='Batch training', ylabel='RMS error, averaged over states', xlabel='Episodes')
    ax.legend()

    plt.show()
