import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.action_selectors import *
from utility.agent import *
from utility.environments import *
from utility.q_approximators import *
from utility.plotter import *

# PARAMETERS
K = 10
NUM_RUNS = 2000
STEPS = 1000
BANDITS = StationaryKArmedBandits(K, mean=0.0, variance=1.0)
PARAMETERS = [
    [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
    [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    [1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    [1/4, 1/2, 1, 2, 4]
]
AGENTS = [
    [
        Agent(K, EpsilonGreedyActionSelector(i), QSampleAvg(K)) for i in PARAMETERS[0]
    ],
    [
        Agent(K, StochasticActionSelector(), GradientBandit(K, i, use_baseline=True)) for i in PARAMETERS[1]
    ],
    [
        Agent(K, UCBActionSelector(K, i), QSampleAvg(K)) for i in PARAMETERS[2]
    ],
    [
        Agent(K, GreedyActionSelector(), QConstantOptimistic(K, 0.1, i)) for i in PARAMETERS[3]
    ]
]
COLOURS = ['r', 'g', 'b', 'k']
LABELS = ['Epsilon', 'Gradient', 'UCB', 'Optimistic']

if __name__ == '__main__':
    parameter_plotter(NUM_RUNS, STEPS, AGENTS, PARAMETERS, BANDITS, COLOURS, LABELS, include_last_x_steps=STEPS)
