import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.policy_iteration import *

class BlackjackDeterministicAgent(object):
    def __call__(self, Q, state):
        hit_value = Q[(state, 0)]
        stick_value = Q[(state, 1)]
        return 0 if hit_value > stick_value else 1

# PARAMETERS
EPISODES = 50000000
ENVIRONMENT = SimplifiedBlackjack()
AGENT = BlackjackDeterministicAgent() # unlike the book our initial policy is arbitrary
GAMMA = 1.0

if __name__ == '__main__':
    fig = plt.figure()
    x_grid, y_grid = np.meshgrid(np.arange(1, 11), np.arange(12, 22))

    Q, _ = monte_carlo_q_exploring_starts(ENVIRONMENT, AGENT, GAMMA, max_iterations=EPISODES)

    # V
    v_array_ace = np.zeros((10, 10))
    v_array_no_ace = np.zeros((10, 10))
    for dealer_card in range(1, 11):
        for player_sum in range(12, 22):
            v_array_ace[dealer_card - 1, player_sum - 12] = Q[((dealer_card, player_sum, True), AGENT(Q, (dealer_card, player_sum, True)))]
            v_array_no_ace[dealer_card - 1, player_sum - 12] = Q[((dealer_card, player_sum, False), AGENT(Q, (dealer_card, player_sum, False)))]

    for id, array, title in zip([2, 4], [v_array_ace, v_array_no_ace], ['V - usable ace', 'V - no usable ace']):
        ax = fig.add_subplot(2, 2, id, projection='3d')
        ax.plot_surface(x_grid, y_grid, array.transpose(), cmap='viridis', edgecolor='none') # notice that v_array needs to be transposed
        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('V')
        ax.set(title=title)

    # PI
    pi_array_ace = np.zeros((10, 10))
    pi_array_no_ace = np.zeros((10, 10))
    for dealer_card in range(1, 11):
        for player_sum in range(12, 22):
            pi_array_ace[player_sum - 12, dealer_card - 1] = AGENT(Q, (dealer_card, player_sum, True))
            pi_array_no_ace[player_sum - 12, dealer_card - 1] = AGENT(Q, (dealer_card, player_sum, False))
    # The figure in the book also displays the action for when the player_sum is 11, even though that is not a "valid" state as
    # previously defined. Therefore I will just stack a row of zeros in front of PI. You will always hit when your sum is 11 anyway.
    pi_array_ace = np.vstack((np.zeros((1, 10)), pi_array_ace))
    pi_array_no_ace = np.vstack((np.zeros((1, 10)), pi_array_no_ace))

    for id, array, title in zip([1, 3], [pi_array_ace, pi_array_no_ace], ['PI - usable ace', 'PI - no usable ace']):
        ax = fig.add_subplot(2, 2, id)
        img = ax.imshow(array, interpolation='none', origin='lower')
        ax.set(title=title)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_yticklabels(['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        cbar = plt.colorbar(img, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Hit', 'Stick'])

    plt.show()
