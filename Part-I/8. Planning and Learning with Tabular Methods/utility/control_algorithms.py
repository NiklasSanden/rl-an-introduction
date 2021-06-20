import numpy as np
from tqdm import tqdm

import random
from collections import defaultdict

def q_learning_update(env, gamma, alpha, Q, state, action, reward, next_state):
    '''
    If next_state is terminal it should be None
    '''
    if next_state:
        actions = env.get_actions(next_state)
        max_q_next_state = np.max(np.array([Q[(next_state, a)] for a in actions]))
        Q[(state, action)] += alpha * (reward + gamma * max_q_next_state - Q[(state, action)])
    else:
        Q[(state, action)] += alpha * (reward - Q[(state, action)])

def dyna_Q(env, agent, gamma, max_episodes, n, alpha=0.1, Q=defaultdict(lambda: 0), model=dict(), model_keys=[], log=True):
    for _ in tqdm(range(max_episodes), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state, Q)
            next_state, reward, terminal, _ = env.step(action)
            if terminal:
                next_state = None
                
            # direct rl
            q_learning_update(env, gamma, alpha, Q, state, action, reward, next_state)

            # update model
            if not ((state, action) in model):
                model_keys.append((state, action))
            model[(state, action)] = (reward, next_state)

            state = next_state

            # planning
            for _ in range(n):
                s, a = random.choice(model_keys)
                r, s_ = model[(s, a)]
                q_learning_update(env, gamma, alpha, Q, s, a, r, s_)

    return (Q, model, model_keys)
