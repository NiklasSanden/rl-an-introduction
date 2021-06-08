import numpy as np
from tqdm import tqdm

from collections import defaultdict

def monte_carlo_v_prediction(env, agent, gamma, max_iterations, start_V=defaultdict(lambda: 0.0), start_N=defaultdict(lambda: 0)):
    '''
    Unlike the pseudo-code in 5.1, this is both every-visit MC and it uses running averages (as described in Ch. 2)
    in order to not keep track of all of the returns.
    The agent should be deterministic and callable with the state as parameter.
    '''
    V = start_V
    N = start_N
    
    for i in tqdm(range(max_iterations)):
        states = [env.reset()]
        actions = []
        rewards = []
        terminal = False
        while not terminal:
            actions.append(agent(states[-1]))
            next_state, reward, terminal, _ = env.step(actions[-1])
            if not terminal:
                states.append(next_state)
            rewards.append(reward)
        assert len(states) == len(actions) == len(rewards), 'The sequence of experience have different lengths'
        G = 0
        for i in reversed(range(len(actions))):
            G = gamma * G + rewards[i]
            N[states[i]] += 1
            V[states[i]] += (1.0 / N[states[i]]) * (G - V[states[i]])

    return (V, N)
