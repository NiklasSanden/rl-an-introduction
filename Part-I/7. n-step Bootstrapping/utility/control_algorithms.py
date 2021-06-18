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

def n_step_sarsa_off_policy(env, agent, behaviour, gamma, max_iterations, n, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent and behaviour should have a method (get_probs) that takes in a state and Q and outputs all of the 
    available actions and the probability of picking each action (pair of lists).
    Since this implementation allows the behaviour policy to change, it calculates the rho_denominator when actions are
    selected and saves the result, so later changes to Q does not alter the probabilities for previous actions. However,
    we can't save the result for the agent since we want it to always use the latest Q when updating, which is why the
    numerator of rho can't be computed iteratively like the denominator can.
    '''
    Q = start_Q
    gammas = calculate_gammas(gamma, n)

    for i in tqdm(range(max_iterations), disable=(not log)):
        states = deque([env.reset()])
        
        behaviour.counter += 1 # add to counter
        _, behaviour_probs = behaviour.get_probs(states[-1], Q)
        action_index = np.random.choice(np.arange(len(behaviour_probs)), p=behaviour_probs)
        rho_denominator = 1.0 # when updating action values the action that is updated should not be adjusted
        b_probs = deque()     # by importance sampling

        action_indices = deque([action_index])
        rewards = deque()
        
        G = 0.0
        terminal = False
        while not terminal or len(states) > 1:
            if not terminal:
                next_state, reward, terminal, _ = env.step(env.get_actions(states[-1])[action_indices[-1]])

                states.append(next_state)
                rewards.append(reward)
                G += gammas[len(rewards) - 1] * reward

                if not terminal:
                    behaviour.counter += 1 # add to counter
                    _, behaviour_probs = behaviour.get_probs(states[-1], Q)
                    action_index = np.random.choice(np.arange(len(behaviour_probs)), p=behaviour_probs)
                    rho_denominator *= behaviour_probs[action_index]
                    b_probs.append(behaviour_probs[action_index])
                    action_indices.append(action_index)
            
            # n_step update
            if len(rewards) == n or terminal:
                rho = 1.0
                if len(b_probs) > 0: # this is only true for the last update (regardless of n)
                    # calculate rho
                    rho_numerator = 1.0
                    states_list = list(states)[1:]           # the action to update should not be  
                    actions_list = list(action_indices)[1:]  # adjusted (see above comment)
                    for a, s in zip(actions_list, states_list):
                        _, agent_probs = agent.get_probs(s, Q)
                        rho_numerator *= agent_probs[a]
                    rho = rho_numerator / rho_denominator
                    b_prob = b_probs.popleft()
                    rho_denominator /= b_prob

                # update
                state_to_update = states.popleft()
                available_actions = env.get_actions(state_to_update)
                action_to_update = available_actions[action_indices.popleft()]

                target = G
                if not terminal:
                    last_action = env.get_actions(states[-1])[action_indices[-1]]
                    target += gammas[n] * Q[(states[-1], last_action)]

                Q[(state_to_update, action_to_update)] += alpha * rho * (target - Q[((state_to_update, action_to_update))])

                first_reward = rewards.popleft()
                G -= first_reward
                G /= gamma

    return Q
