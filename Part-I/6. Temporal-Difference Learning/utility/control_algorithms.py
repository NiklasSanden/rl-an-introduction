import numpy as np
from tqdm import tqdm

from collections import defaultdict

def sarsa_on_policy_td_q(env, agent, gamma, max_iterations, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state and Q-function as parameters and return one action.
    '''
    Q = start_Q

    for i in tqdm(range(max_iterations), disable=(not log)):
        state = env.reset()
        action = agent(state, Q)
        terminal = False
        while not terminal:
            next_state, reward, terminal, _ = env.step(action)
            if terminal: # just in case the user gave a start_Q where Q(terminal, a) != 0
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
            else:
                next_action = agent(next_state, Q)
                Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])
                action = next_action
            state = next_state

    return Q

def q_learning(env, agent, gamma, max_iterations, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state and Q-function as parameters and return one action.
    '''
    Q = start_Q

    for i in tqdm(range(max_iterations), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state, Q)
            next_state, reward, terminal, _ = env.step(action)
            if terminal: # just in case the user gave a start_Q where Q(terminal, a) != 0
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
            else:
                actions = env.get_actions(next_state)
                max_q_next_state = np.max(np.array([Q[(next_state, a)] for a in actions]))
                Q[(state, action)] += alpha * (reward + gamma * max_q_next_state - Q[(state, action)])
            state = next_state

    return Q

def expected_sarsa_on_policy(env, agent, gamma, max_iterations, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state and Q-function as parameters and return one action.
    It should also have a method (get_probs) that takes in a state and Q and outputs all of the 
    available actions and the probability of picking each action (pair of lists).
    '''
    Q = start_Q

    for i in tqdm(range(max_iterations), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state, Q)
            next_state, reward, terminal, _ = env.step(action)
            if terminal: # just in case the user gave a start_Q where Q(terminal, a) != 0
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
            else:
                actions, probs = agent.get_probs(next_state, Q)
                expected_next_value = np.sum(np.array(probs) * np.array([Q[(next_state, a)] for a in actions]))
                Q[(state, action)] += alpha * (reward + gamma * expected_next_value - Q[(state, action)])
            state = next_state

    return Q
