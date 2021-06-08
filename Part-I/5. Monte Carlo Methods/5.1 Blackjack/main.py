import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_iteration import *

class TrivialAgent(object):
    def __init__(self, stick_after=20):
        self.stick_after = stick_after

    def __call__(self, state):
        _, player_sum, _ = state
        return 0 if player_sum < self.stick_after else 1

def plot(id, array, title, fig):
    x_grid, y_grid = np.meshgrid(np.arange(1, 11), np.arange(12, 22))
    ax = fig.add_subplot(2, 2, id, projection='3d')
    ax.plot_surface(x_grid, y_grid, array.transpose()) # notice that array needs to be transposed
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V')
    ax.set(title=title)

def get_v_arrays(V):
    v_array_ace = np.zeros((10, 10))
    v_array_no_ace = np.zeros((10, 10))
    for dealer_card in range(1, 11):
        for player_sum in range(12, 22):
            v_array_ace[dealer_card - 1, player_sum - 12] = V[(dealer_card, player_sum, True)]
            v_array_no_ace[dealer_card - 1, player_sum - 12] = V[(dealer_card, player_sum, False)]
    return (v_array_ace, v_array_no_ace)

# PARAMETERS
EPISODES_1 = 10000
EPISODES_2 = 500000
ENVIRONMENT = SimplifiedBlackjack()
AGENT = TrivialAgent(stick_after=20)
GAMMA = 1.0

if __name__ == '__main__':
    fig = plt.figure()
    # EPISODES_1
    V, N = monte_carlo_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=EPISODES_1)
    v_array_ace, v_array_no_ace = get_v_arrays(V)
    plot(1, v_array_ace, 'Usable ace - ' + str(EPISODES_1) + ' episodes', fig)
    plot(3, v_array_no_ace, 'No usable ace - ' + str(EPISODES_1) + ' episodes', fig)

    # EPISODES_2
    V, N = monte_carlo_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=EPISODES_2 - EPISODES_1, start_V=V, start_N=N)
    v_array_ace, v_array_no_ace = get_v_arrays(V)
    plot(2, v_array_ace, 'Usable ace - ' + str(EPISODES_2) + ' episodes', fig)
    plot(4, v_array_no_ace, 'No usable ace - ' + str(EPISODES_2) + ' episodes', fig)

    plt.show()
