import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
NUM_RUNS = 10
NUM_DOUBLES = 5 # 8 in the book
ENVIRONMENT = DynaMaze()
DYNA_Q_AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=0.5)
PRIO_AGENT = EpsilonGreedyAgent(ENVIRONMENT, epsilon=1.0)
N = 5
THETA = 1e-9
GAMMA = 0.95
ALPHA = 1.0
COLOURS = ['blue', 'red']
LABELS = ['Dyna-Q', 'Prioritized sweeping']

def double_env_res(env):
    blocks = {(row * 2, col) for row, col in env.blocks}
    blocks_odd = {(row * 2 + 1, col) for row, col in env.blocks}
    return DynaMaze(env.rows * 2, env.cols, (env.start[0] * 2, env.start[1]), env.end, blocks.union(blocks_odd))

def get_best_path():
    counter = 0
    visited = set()
    states = [ENVIRONMENT.reset()]
    while states:
        next_states = []
        for s in states:
            if s in visited:
                continue
            visited.add(s)

            if s == ENVIRONMENT.end:
                return counter
            
            for a in ENVIRONMENT.get_actions(None):
                next_state = (s[0] + a[0], s[1] + a[1])
                if ENVIRONMENT._is_invalid_state(next_state):
                    continue
                next_states.append(next_state)
        counter += 1
        states = next_states
        next_states = []
    raise Exception('No path found')

def is_optimal(Q, optimal):
    greedy = EpsilonGreedyAgent(ENVIRONMENT, epsilon=-1.0)
    state = ENVIRONMENT.reset()
    for i in range(1, optimal + 2):
        state, _, terminal, _ = ENVIRONMENT.step(greedy(state, Q))
        if terminal:
            break
    assert i >= optimal, 'the algorithm claims to have done better than optimal'
    return i == optimal

if __name__ == '__main__':
    fig = plt.figure()

    updates = np.zeros((NUM_RUNS, NUM_DOUBLES, len(LABELS)))
    for d in tqdm(range(NUM_DOUBLES), disable=False):
        optimal_steps = get_best_path()
        for run in tqdm(range(NUM_RUNS), leave=False, disable=False):
            # Dyna-Q
            Q, model, model_keys = (defaultdict(lambda: 0), dict(), [])
            updater = QUpdater()
            optimal = False
            while not optimal:
                Q, model, model_keys = dyna_Q(ENVIRONMENT, DYNA_Q_AGENT, GAMMA, max_episodes=1, n=N, alpha=ALPHA, Q=Q, model=model, 
                                              model_keys=model_keys, q_updater=updater, log=False)
                optimal = is_optimal(Q, optimal_steps)
            updates[run, d, 0] = updater.counter

            # Priority Sweep
            Q, model, reverse_model, queue = (defaultdict(lambda: 0), dict(), defaultdict(lambda: set()), PrioritizedSweepingQueue())
            updater = QUpdater()
            optimal = False
            while not optimal:
                Q, model, reverse_model, queue = dyna_Q_prioritized_sweeping(ENVIRONMENT, PRIO_AGENT, GAMMA, max_episodes=1, n=N, theta=THETA, alpha=ALPHA, Q=Q, 
                                                                             model=model, reverse_model=reverse_model, queue=queue, q_updater=updater, log=False)
                optimal = is_optimal(Q, optimal_steps)
            updates[run, d, 1] = updater.counter
        ENVIRONMENT = double_env_res(ENVIRONMENT)

    updates = np.average(updates, axis=0)

    ax = fig.add_subplot()
    X = [47 * (2 ** x) for x in range(0, NUM_DOUBLES)]
    for a in range(len(LABELS)):
        ax.plot(X, updates[:, a], color=COLOURS[a], label=LABELS[a])
    ax.set(ylabel='Updates until optimal solution', xlabel='Gridworld size (#states)')
    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=2)
    ax.set_xticks(X)
    ax.set_xticklabels(X)
    ax.legend()

    plt.show()
