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
GRAPH_ON_EPISODES = [1, 12, 104, 300, 1000, 9000]
ENVIRONMENT = MountainCar(min_x=-1.2, max_x=0.5, min_v=-0.07, max_v=0.07, speed=0.001, gravity=0.0025, freq=3, 
                          start_x_min=-0.6, start_x_max=-0.4, start_v_min=0, start_v_max=0)
AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.0) # we can be greedy since the initial values are optimistic
Q = TileCoding(num_tilings=8, dim_sizes=[ENVIRONMENT.max_x - ENVIRONMENT.min_x, ENVIRONMENT.max_v - ENVIRONMENT.min_v], max_size=4096)
GAMMA = 1
ALPHA = 0.3
GRAPH_DIMS = [2, 3]
GRAPH_RESOLUTION = 30

if __name__ == '__main__':
    fig = plt.figure()
    x_graph_points = np.linspace(ENVIRONMENT.min_x, ENVIRONMENT.max_x, num=GRAPH_RESOLUTION)
    velocity_graph_points = np.linspace(ENVIRONMENT.min_v, ENVIRONMENT.max_v, num=GRAPH_RESOLUTION)
    x_grid, y_grid = np.meshgrid(x_graph_points, velocity_graph_points)

    episode = 0
    for (idx, e) in enumerate(GRAPH_ON_EPISODES):
        Q = semi_gradient_sarsa(ENVIRONMENT, AGENT, GAMMA, max_episodes=e - episode, alpha=ALPHA, Q=Q, log=True)
        episode = e
        
        v_array = np.zeros((GRAPH_RESOLUTION, GRAPH_RESOLUTION))
        for x in range(GRAPH_RESOLUTION):
            for y in range(GRAPH_RESOLUTION):
                state = (x_graph_points[x], velocity_graph_points[y])
                actions = ENVIRONMENT.get_actions(state)
                best_value = np.max([Q(state, a) for a in actions])
                v_array[y, x] = -best_value

        ax = fig.add_subplot(GRAPH_DIMS[0], GRAPH_DIMS[1], idx + 1, projection='3d')
        ax.plot_surface(x_grid, y_grid, v_array, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Cost to go')
        ax.set(title=f'Episode {e}')

    plt.show()
