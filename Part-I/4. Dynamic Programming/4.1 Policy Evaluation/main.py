import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.dynamics import *
from utility.policy_evaluation import *

# PARAMETERS
N = 4
M = 4
TERMINALS = {(0, 0), (N-1, M-1)}
PI = lambda s: [1/4, 1/4, 1/4, 1/4]
MAX_DELTA = 0.0001
GAMMA = 1.0

if __name__ == '__main__':
    p = GridworldNxM(N, M)
    V, _ = v_evaluation(PI, p, TERMINALS, MAX_DELTA, GAMMA)
    for n in range(N):
        print([round(V[(n, m)], 1) for m in range(M)])
