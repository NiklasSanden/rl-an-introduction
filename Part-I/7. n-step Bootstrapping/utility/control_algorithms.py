import numpy as np
from tqdm import tqdm

from .misc import calculate_gammas

from collections import defaultdict
from collections import deque

def n_step_sarsa(env, agent, gamma, max_iterations, n, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state and Q-function as parameters and return one action.
    '''
    Q = start_Q
    gammas = calculate_gammas(gamma, n)

    for i in tqdm(range(max_iterations), disable=(not log)):
        states = deque([env.reset()])
        actions = deque([agent(states[-1], Q)])
        rewards = deque()
        G = 0.0
        terminal = False
        while not terminal or len(states) > 1:
            if not terminal:
                next_state, reward, terminal, _ = env.step(actions[-1])

                states.append(next_state)
                rewards.append(reward)
                G += gammas[len(rewards) - 1] * reward

                if not terminal:
                    actions.append(agent(states[-1], Q))
            
            # n_step update
            if len(rewards) == n or terminal:
                state_to_update = states.popleft()
                action_to_update = actions.popleft()

                target = G if terminal else G + gammas[n] * Q[(states[-1], actions[-1])]
                Q[(state_to_update, action_to_update)] += alpha * (target - Q[((state_to_update, action_to_update))])

                first_reward = rewards.popleft()
                G -= first_reward
                G /= gamma

    return Q

def n_step_expected_sarsa_on_policy(env, agent, gamma, max_iterations, n, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state and Q-function as parameters and return one action.
    It should also have a method (get_probs) that takes in a state and Q and outputs all of the 
    available actions and the probability of picking each action (pair of lists).
    '''
    Q = start_Q
    gammas = calculate_gammas(gamma, n)

    for i in tqdm(range(max_iterations), disable=(not log)):
        states = deque([env.reset()])
        actions = deque()
        rewards = deque()
        G = 0.0
        terminal = False
        while not terminal or len(states) > 1:
            if not terminal:
                actions.append(agent(states[-1], Q))
                next_state, reward, terminal, _ = env.step(actions[-1])

                states.append(next_state)
                rewards.append(reward)
                G += gammas[len(rewards) - 1] * reward

            # n_step update
            if len(rewards) == n or terminal:
                state_to_update = states.popleft()
                action_to_update = actions.popleft()

                target = G
                if not terminal:
                    next_actions, probs = agent.get_probs(states[-1], Q)
                    expected_next_value = np.sum(np.array(probs) * np.array([Q[(states[-1], a)] for a in next_actions]))
                    target += gammas[n] * expected_next_value
                Q[(state_to_update, action_to_update)] += alpha * (target - Q[((state_to_update, action_to_update))])

                first_reward = rewards.popleft()
                G -= first_reward
                G /= gamma
            
    return Q
