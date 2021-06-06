import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.dynamics import *
from utility.policy_evaluation import *

# PARAMETERS
MAX_DELTA = 0.01
GAMMA = 0.9
MAX_A = 20
MAX_B = 20
MAX_TRANSFER = 5
P = JacksCarRental(max_a=MAX_A, max_b=MAX_B, max_transfer=MAX_TRANSFER, extended=True, cut_random_loop=10)
TERMINALS = {}
PI = lambda s: { action : 1.0 if action == 0 else 0.0 for action in range(-MAX_TRANSFER, MAX_TRANSFER + 1) }

if __name__ == '__main__':
    V, optimal = v_evaluation(PI, P, TERMINALS, MAX_DELTA, GAMMA)
    while not optimal:
        PI = get_greedy_PI_from_V(V, P, GAMMA)
        V, optimal = v_evaluation(PI, P, TERMINALS, MAX_DELTA, GAMMA, V)
    print('Optimal')

    # PI
    pi_array = np.zeros((MAX_A + 1, MAX_B + 1), dtype=int)
    for state_a in range(MAX_A + 1):
        for state_b in range(MAX_B + 1):
            best_action = -1
            best_value = float('-inf')
            for key in PI((state_a, state_b)):
                if PI((state_a, state_b))[key] > best_value:
                    best_value = PI((state_a, state_b))[key]
                    best_action = key
            pi_array[state_a, state_b] = best_action
    
    # V
    v_array = np.zeros((MAX_A + 1, MAX_B + 1))
    for state_a in range(MAX_A + 1):
        for state_b in range(MAX_B + 1):
            v_array[state_a, state_b] = V[(state_a, state_b)]
    x_grid, y_grid = np.meshgrid(np.arange(MAX_A + 1), np.arange(MAX_B + 1))
    
    # Plot PI
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    img = ax.imshow(pi_array, interpolation='none', origin='lower')
    plt.colorbar(img, ax=ax)
    ax.set(title='PI', ylabel='Average reward', xlabel='Steps')

    # Plot V
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x_grid, y_grid, v_array,cmap='viridis', edgecolor='none')
    ax.set_xlabel('#Cars at first location')
    ax.set_ylabel('#Cars at second location')
    ax.set_zlabel('V under PI')
    plt.show()
