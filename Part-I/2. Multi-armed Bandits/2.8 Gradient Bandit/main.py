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
BANDITS = StationaryKArmedBandits(K, mean=4.0, variance=1.0)
AGENTS = [
    Agent(K, StochasticActionSelector(),        GradientBandit(K, 0.1, use_baseline=True)),
    Agent(K, StochasticActionSelector(),        GradientBandit(K, 0.1, use_baseline=False)),
    Agent(K, UCBActionSelector(K, 2.0),         QSampleAvg(K)),
    Agent(K, EpsilonGreedyActionSelector(0.1),  QConstant(K, 0.1))
]
COLOURS = ['r', 'g', 'b', 'm']
LABELS = ['Gradient baseline', 'Gradient no baseline', 'UCB', 'Constant Epsilon Greedy']

if __name__ == '__main__':
    plotter(NUM_RUNS, STEPS, AGENTS, BANDITS, COLOURS, LABELS)
