import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.policy_evaluation import *
from utility.agents import *
from utility.function_approximators import *

# PARAMETERS
NUM_RUNS = 30
EPISODES = 5000
EPISODE_PLOT_INTERVAL = 50
EPISODES_TO_GET_STATE_DISTRIBUTION = 100000
NUM_STATES = 1000
JUMP_SIZE = 100
ENVIRONMENT = RandomWalk(num_states=NUM_STATES, step_size=JUMP_SIZE)
AGENT = RepeatableWrapper(StateCounterWrapper(UniformAgent(ENVIRONMENT)))
NUM_TILINGS = 50
V = [TileCoding(num_tilings=1, width=200, highest_value=NUM_STATES), TileCoding(num_tilings=NUM_TILINGS, width=200, highest_value=NUM_STATES)]
MAX_DELTA = 0.0001
GAMMA = 1
ALPHA = 0.0001
COLOURS = ['red', 'green']
LABELS = ['State aggregation (one tiling)', 'Tile coding (' + str(NUM_TILINGS)  + ' tilings)']

if __name__ == '__main__':
    fig = plt.figure()

    states = [s for s in range(1, NUM_STATES + 1)]

    # Get true values and state distribution
    true_values = ENVIRONMENT.get_true_value(GAMMA, MAX_DELTA)
    distribution = get_state_distribution(ENVIRONMENT, AGENT, EPISODES_TO_GET_STATE_DISTRIBUTION, states=states)

    # Get values
    errors = np.zeros((NUM_RUNS, EPISODES // EPISODE_PLOT_INTERVAL, len(LABELS)))
    for run in tqdm(range(NUM_RUNS)):
        AGENT.reset()
        for a in tqdm(range(len(LABELS)), leave=False):
            V[a].zero_weights()
            AGENT.repeat()
            for e in tqdm(range(EPISODES), leave=False):
                V[a] = gradient_monte_carlo_v(ENVIRONMENT, AGENT, GAMMA, max_episodes=1, alpha=ALPHA, V=V[a], log=False)
                if e % EPISODE_PLOT_INTERVAL == 0:
                    errors[run, e // EPISODE_PLOT_INTERVAL, a] = np.sqrt(calculate_VE(lambda s: true_values[s - 1], V[a], states, distribution))
    
    errors = np.average(errors, axis=0)

    ax = fig.add_subplot()
    for a in range(len(LABELS)):
        ax.plot(np.arange(EPISODES // EPISODE_PLOT_INTERVAL) * EPISODE_PLOT_INTERVAL, errors[:, a], color=COLOURS[a], label=LABELS[a])
    ax.set(ylabel='sqrt(VE) averaged over ' + str(NUM_RUNS) + ' runs', xlabel='Episodes')
    ax.legend()

    plt.show()
