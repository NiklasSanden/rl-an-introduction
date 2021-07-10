from tqdm import tqdm

from .misc import *

def semi_gradient_sarsa(env, agent, gamma, max_episodes, alpha, Q, log=True):
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        state = env.reset()
        action = agent(state, Q)
        terminal = False
        while not terminal:
            next_state, reward, terminal, _ = env.step(action)
            
            if terminal:
                change = (reward - Q(state, action)) * Q.get_gradients(state, action)
                Q.update_weights(alpha, change)
                break
                
            next_action = agent(next_state, Q)
            change = (reward + gamma * Q(next_state, next_action) - Q(state, action)) * Q.get_gradients(state, action)
            Q.update_weights(alpha, change)

            state = next_state
            action = next_action
    return Q

def semi_gradient_n_step_sarsa(env, agent, gamma, n, max_episodes, alpha, Q, log=True):
    gammas = calculate_gammas(gamma, n)

    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        states = CircularList(n + 1, env.reset())
        rewards = CircularList(n + 1, 0)
        actions = CircularList(n + 1, agent(states[0], Q))
        t = 0
        T = float('inf')
        tau = 0
        G = 0.0
        while tau != T - 1:
            if t < T:
                states[t + 1], rewards[t + 1], terminal, _ = env.step(actions[t])
                G += gammas[min(t, n)] * rewards[t + 1]
                
                if terminal:
                    T = t + 1
                else:
                    actions[t + 1] = agent(states[t + 1], Q)
            
            tau = t - n + 1
            if tau >= 0:
                n_return = G + gammas[n] * Q(states[tau + n], actions[tau + n]) if tau + n < T else G
                change = (n_return - Q(states[tau], actions[tau])) * Q.get_gradients(states[tau], actions[tau])
                Q.update_weights(alpha, change)

                G -= rewards[tau + 1]
                G /= gamma
            
            t += 1
    return Q
