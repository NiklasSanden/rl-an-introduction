import numpy as np
from tqdm import tqdm

from .misc import calculate_gammas

from collections import defaultdict
from collections import deque

def n_step_TD(env, agent, gamma, max_iterations, n, alpha=0.1, start_V=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state as parameter and return one action.
    '''
    V = start_V
    gammas = calculate_gammas(gamma, n)
    
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

def n_step_sum_of_TD_errors(env, agent, gamma, max_iterations, n, use_latest_V_for_TD=False, alpha=0.1, start_V=defaultdict(lambda: 0.0), log=True):
    '''
    If use_latest_V_for_TD is True, then the TD-errors for one update will use the same V, meaning that it is the same as n_step as proven
    in exercise 7.1. If it is False, then delta_t will use V_t and it won't be the same as V, similar to exercise 6.1.
    use_latest_V_for_TD being False is used to answer exercise 7.2.
    The agent should be callable with the state as parameter and return one action.
    '''
    V = start_V
    gammas = calculate_gammas(gamma, n)
    
    for i in tqdm(range(max_iterations), disable=(not log)):
        states = [env.reset()]
        rewards = []
        V_copies = []
        terminal = False
        while not terminal or len(states) > 1:
            if use_latest_V_for_TD:
                V_copies.append(V) # use a reference instead of copying so that the latest V is always used
            else:
                V_copies.append(V.copy())

            if not terminal:
                action = agent(states[-1])
                next_state, reward, terminal, _ = env.step(action)

                states.append(next_state)
                rewards.append(reward)
            
            # n_step update
            if len(rewards) == n or terminal:
                state_to_update = states[0]
                td_sum = np.sum(np.array([g * (r + gamma * V_copy[s_] - V_copy[s]) for g, r, s_, s, V_copy in zip(gammas[:-1], rewards, states[1:], states[:-1], V_copies)]))
                V[state_to_update] += alpha * td_sum

                states.pop(0)
                rewards.pop(0)
                V_copies.pop(0)

    return V

def n_step_sum_of_TD_errors_iterative(env, agent, gamma, max_iterations, n, alpha=0.1, start_V=defaultdict(lambda: 0.0), log=True):
    '''
    This is an iterative implementation of n_step_sum_of_TD_errors that uses deques, making it significantly more efficient,
    especially for higher ns. The cost of this is not being able to support use_latest_V_for_TD to turn it into n_step.
    The agent should be callable with the state as parameter and return one action.
    '''
    V = start_V
    gammas = calculate_gammas(gamma, n)
    
    for i in tqdm(range(max_iterations), disable=(not log)):
        states = deque([env.reset()])
        td_errors = deque()
        G = 0.0
        terminal = False
        while not terminal or len(states) > 1:
            if not terminal:
                action = agent(states[-1])
                next_state, reward, terminal, _ = env.step(action)

                td_errors.append(reward + gamma * V[next_state] - V[states[-1]])
                G += gammas[len(td_errors) - 1] * td_errors[-1]
                states.append(next_state) # make sure this is after td_errors.append since it uses states[-1]
            
            # n_step update
            if len(td_errors) == n or terminal:
                state_to_update = states.popleft()
                V[state_to_update] += alpha * G
                
                first_error = td_errors.popleft()
                G -= first_error
                G /= gamma

    return V
