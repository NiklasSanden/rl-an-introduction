import numpy as np
from tqdm import tqdm

from collections import defaultdict
from collections import deque

def n_step_TD(env, agent, gamma, max_iterations, n, alpha=0.1, start_V=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state as parameter and return one action.
    '''
    V = start_V

    gammas = np.ones(n + 1) # precalculate all gammas ** i
    for i in range(1, len(gammas)):
        gammas[i] = gammas[i - 1] * gamma
    
    for i in tqdm(range(max_iterations), disable=(not log)):
        states = deque([env.reset()])
        rewards = deque()
        G = 0.0
        terminal = False
        while not terminal or len(states) > 1:
            if not terminal:
                action = agent(states[-1])
                next_state, reward, terminal, _ = env.step(action)

                states.append(next_state)
                rewards.append(reward)
                G += gammas[len(rewards) - 1] * reward
            
            # n_step update
            if len(rewards) == n or terminal:
                state_to_update = states.popleft()
                bootstrap_state = states[-1]
                V[state_to_update] += alpha * (G + gammas[n] * V[bootstrap_state] - V[state_to_update])
                
                first_reward = rewards.popleft()
                G -= first_reward
                G /= gamma

    return V
