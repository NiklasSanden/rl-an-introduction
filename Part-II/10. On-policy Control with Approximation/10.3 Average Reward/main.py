import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.agents import *
from utility.control_algorithms import *
from utility.environments import *
from utility.function_approximators import *
from utility.misc import *

# PARAMETERS
ITERATIONS = 2000000
MAX_SERVERS = 10            # The plotting code currently
PRIORITIES = [1, 2, 4, 8]   # hard-codes these values
ENVIRONMENT = ServerAssignment(max_servers=MAX_SERVERS, priorities=PRIORITIES, free_server_p=0.06)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.1)
Q = Tabular(num_states=(MAX_SERVERS + 1) * len(PRIORITIES), num_actions=len(ENVIRONMENT.get_actions(None)))
ALPHA = 0.01
BETA = 0.01
COLOURS = ['red', 'green', 'blue', 'black']

if __name__ == '__main__':
    fig = plt.figure()
    
    Q, R_bar = differential_semi_gradient_sarsa(ENVIRONMENT, AGENT, ITERATIONS, ALPHA, BETA, Q, R_bar=0, log=True)

    # Show actions
    ax = fig.add_subplot(2, 1, 1)
    actions = np.zeros((len(PRIORITIES), MAX_SERVERS), dtype=int)
    for p in range(len(PRIORITIES)):
        for s in range(MAX_SERVERS):
            actions[p, s] = 1 if Q((s + 1, PRIORITIES[p]), 1) >= Q((s + 1, PRIORITIES[p]), 0) else 0
    img = ax.imshow(actions, interpolation='none')#, origin='lower')
    ax.set(title='Policy', xlabel='Number of free servers', ylabel='Priority')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['1', '2', '4', '8'])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    cbar = plt.colorbar(img, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['REJECT', 'ACCEPT'])

    # Show value function
    ax = fig.add_subplot(2, 1, 2)
    for p in range(len(PRIORITIES)):
        values = [max(Q((s, PRIORITIES[p]), 0), Q((s, PRIORITIES[p]), 1)) for s in range(MAX_SERVERS + 1)]
        ax.plot(values, color=COLOURS[p], label=f'priority {PRIORITIES[p]}')
    ax.set(ylabel='Differential value of best action', xlabel='Number of free servers')
    ax.legend()

    # Print the learned R_bar
    print(f'The learned R_bar = {R_bar}')

    plt.show()
