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

# PARAMETERS
EPISODES_1 = 10000
EPISODES_2 = 500000
ENVIRONMENT = SimplifiedBlackjack()
AGENT = TrivialAgent(stick_after=20)
GAMMA = 1.0

if __name__ == '__main__':
    # EPISODES_1
    V = monte_carlo_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=EPISODES_1)

    v_array_ace = np.zeros((10, 10))
    v_array_no_ace = np.zeros((10, 10))
    for dealer_card in range(1, 11):
        for player_sum in range(12, 22):
            v_array_ace[dealer_card - 1, player_sum - 12] = V[(dealer_card, player_sum, True)]
            v_array_no_ace[dealer_card - 1, player_sum - 12] = V[(dealer_card, player_sum, False)]
    x_grid, y_grid = np.meshgrid(np.arange(1, 11), np.arange(12, 22))

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(x_grid, y_grid, v_array_ace.transpose()) # notice that v_array needs to be transposed
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V')
    ax.set(title='Usable ace - ' + str(EPISODES_1) + ' episodes')

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.plot_surface(x_grid, y_grid, v_array_no_ace.transpose())
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V')
    ax.set(title='No usable ace - ' + str(EPISODES_1) + ' episodes')

    # EPISODES_2
    V = monte_carlo_v_prediction(ENVIRONMENT, AGENT, GAMMA, max_iterations=EPISODES_2 - EPISODES_1, start_V=V)

    v_array_ace = np.zeros((10, 10))
    v_array_no_ace = np.zeros((10, 10))
    for dealer_card in range(1, 11):
        for player_sum in range(12, 22):
            v_array_ace[dealer_card - 1, player_sum - 12] = V[(dealer_card, player_sum, True)]
            v_array_no_ace[dealer_card - 1, player_sum - 12] = V[(dealer_card, player_sum, False)]
    x_grid, y_grid = np.meshgrid(np.arange(1, 11), np.arange(12, 22))

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot_surface(x_grid, y_grid, v_array_ace.transpose()) # notice that v_array needs to be transposed
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V')
    ax.set(title='Usable ace - ' + str(EPISODES_2) + ' episodes')

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(x_grid, y_grid, v_array_no_ace.transpose())
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V')
    ax.set(title='No usable ace - ' + str(EPISODES_2) + ' episodes')

    plt.show()
