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
    '''
    Q = start_Q
    gammas = calculate_gammas(gamma, n)

    for i in tqdm(range(max_iterations), disable=(not log)):
        states = deque([env.reset()])
        
        behaviour.counter += 1 # add to counter
        _, behaviour_probs = behaviour.get_probs(states[-1], Q)
        action_index = np.random.choice(np.arange(len(behaviour_probs)), p=behaviour_probs)
        target_probs = deque()          # when updating action values the action
        rho_denominator = 1.0           # that is updated should not be adjusted
        rho_denominator_probs = deque() # by importance sampling

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
                    _, agent_probs = agent.get_probs(states[-1], Q)
                    action_index = np.random.choice(np.arange(len(behaviour_probs)), p=behaviour_probs)
                    target_probs.append(agent_probs[action_index])
                    rho_denominator *= behaviour_probs[action_index]
                    rho_denominator_probs.append(behaviour_probs[action_index])
                    action_indices.append(action_index)
            
            # n_step update
            if len(rewards) == n or terminal:
                rho = 1.0
                if len(rho_denominator_probs) > 0: # this is only true for the last update (regardless of n)
                    rho = np.prod(np.array(target_probs)) / rho_denominator
                    target_probs.popleft()
                    first_rho_prob = rho_denominator_probs.popleft()
                    rho_denominator /= first_rho_prob

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

def n_step_sarsa_off_policy_control_variate(env, agent, behaviour, gamma, max_iterations, n, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    return n_step_sarsa_off_policy_Q_sigma(env, agent, behaviour, gamma, max_iterations, n, sigma=1, alpha=alpha, start_Q=start_Q, log=log)

def n_step_sarsa_off_policy_tree_backup(env, agent, behaviour, gamma, max_iterations, n, sigma, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    return n_step_sarsa_off_policy_Q_sigma(env, agent, behaviour, gamma, max_iterations, n, sigma=0, alpha=alpha, start_Q=start_Q, log=log)

def n_step_sarsa_off_policy_Q_sigma(env, agent, behaviour, gamma, max_iterations, n, sigma, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent and behaviour should have a method (get_probs) that takes in a state and Q and outputs all of the 
    available actions and the probability of picking each action (pair of lists).
    This only supports one value of sigma for each choice.
    '''
    Q = start_Q

    for i in tqdm(range(max_iterations), disable=(not log)):
        states = [env.reset()]
        
        behaviour.counter += 1 # add to counter
        available_actions, behaviour_probs = behaviour.get_probs(states[-1], Q)
        action_index = np.random.choice(np.arange(len(behaviour_probs)), p=behaviour_probs)
        action_indices = [action_index]
        actions = [available_actions[action_index]]
        
        rewards = [0]   # dummy reward for the sake of the first one being t + 1
        rho_probs = [0] # another dummy value since the first rho is at t + 1 when updating action values
        
        T = float('inf')
        tau = -1
        t = 0
        while tau != T - 1:
            if t < T:
                next_state, reward, terminal, _ = env.step(actions[-1])

                states.append(next_state)
                rewards.append(reward)

                if terminal:
                    T = t + 1
                else:
                    behaviour.counter += 1 # add to counter
                    available_actions, behaviour_probs = behaviour.get_probs(states[-1], Q)
                    _, pi_probs = agent.get_probs(states[-1], Q)
                    action_index = np.random.choice(np.arange(len(behaviour_probs)), p=behaviour_probs)
                    rho_probs.append(pi_probs[action_index] / behaviour_probs[action_index])
                    action_indices.append(action_index)
                    actions.append(available_actions[action_index])

            tau = t - n + 1
            
            # n_step update
            if tau >= 0:
                if t + 1 < T:
                    G = Q[(states[t + 1], actions[t + 1])]
                for k in range(min(t + 1, T), (tau + 1) - 1, -1): # (tau + 1) - 1 to go through tau + 1
                    if k == T:
                        G = rewards[T]
                    else:
                        available_actions, agent_probs = agent.get_probs(states[k], Q)
                        V_approx = np.sum(np.array(agent_probs) * np.array([Q[(states[k], a)] for a in available_actions]))
                        G = rewards[k] + gamma * (sigma * rho_probs[k] + (1 - sigma) * agent_probs[action_indices[k]]) * (G - Q[(states[k], actions[k])]) + gamma * V_approx

                Q[(states[tau], actions[tau])] += alpha * (G - Q[(states[tau], actions[tau])])
            
            t += 1

    return Q
