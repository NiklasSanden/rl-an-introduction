import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.dynamics import *
from utility.policy_evaluation import *

# PARAMETERS
GOAL = 100
P = GamblersProblem(p=0.45, goal=GOAL)
TERMINALS = {0, GOAL}
MAX_DELTA = 0.0
GAMMA = 1.0
ITERATIONS_TO_PLOT = [1, 2, 3, 32, float('inf')]
COLOURS = ['g', 'r', 'b', 'm', 'y']
LABELS = ["sweep 1", "sweep 2", "sweep 3", "sweep 32", "Final v"]

def plot_value_function(X, V, label, colour):
    Y = [V[state] for state in X]
    plt.plot(X, Y, label=label, color=colour)

if __name__ == "__main__":
    X = [s for s in range(1, GOAL)]

    for idx, i in enumerate(ITERATIONS_TO_PLOT):
        print("Testing " + LABELS[idx])
        V = v_value_iteration(P, TERMINALS, MAX_DELTA, GAMMA, max_iterations=i)
        plot_value_function(X, V, LABELS[idx], COLOURS[idx])
    plt.xlabel("Capital")
    plt.ylabel("Value estimates")
    plt.legend()
    plt.show()

    PI = get_greedy_PI_from_V(V, P, GAMMA, should_round=True)
    Y = [get_most_likely_action(PI, s) for s in X]
    plt.bar(X, Y)
    plt.xlabel("Capital")
    plt.ylabel("Final policy (stake)")
    plt.show()
