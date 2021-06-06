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
AGENTS = [
    Agent(K, GreedyActionSelector(),            QSampleAvg(K)),
    Agent(K, EpsilonGreedyActionSelector(0.01), QSampleAvg(K)),
    Agent(K, EpsilonGreedyActionSelector(0.1),  QSampleAvg(K))
]
COLOURS = ['g', 'r', 'b']
LABELS = ['eps=0.0', 'eps=0.01', 'eps=0.1']

if __name__ == '__main__':
    plotter(NUM_RUNS, STEPS, AGENTS, BANDITS, COLOURS, LABELS)
