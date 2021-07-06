import numpy as np
from tqdm import tqdm

from .misc import *

def gradient_monte_carlo_v(env, agent, gamma, max_episodes, alpha, V, log=True):
    for i in tqdm(range(max_episodes), disable=(not log)):
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
            change = (G - V(states[i])) * V.get_gradients(states[i])
            V.update_weights(alpha, change)
    return V

def semi_gradient_TD_0_v(env, agent, gamma, max_episodes, alpha, V, log=True):
    for i in tqdm(range(max_episodes), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state)
            next_state, reward, terminal, _ = env.step(action)
            
            # update V
            if terminal:
                change = (reward - V(state)) * V.get_gradients(state)
            else:
                change = (reward + gamma * V(next_state) - V(state)) * V.get_gradients(state)
            V.update_weights(alpha, change)

            state = next_state
    return V

def semi_gradient_n_step_TD_v(env, agent, gamma, max_episodes, alpha, n, V, log=True):
    gammas = calculate_gammas(gamma, n)

    for i in tqdm(range(max_episodes), disable=(not log)):
        states = CircularList(n + 1, env.reset())
        rewards = CircularList(n + 1, 0)
        t = 0
        T = float('inf')
        tau = 0
        G = 0.0
        while tau != T - 1:
            if t < T:
                action = agent(states[t])
                states[t + 1], rewards[t + 1], terminal, _ = env.step(action)
                G += gammas[min(t, n)] * rewards[t + 1]
                if terminal:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                n_return = G + gammas[n] * V(states[tau + n]) if tau + n < T else G
                change = (n_return - V(states[tau])) * V.get_gradients(states[tau])
                V.update_weights(alpha, change)

                G -= rewards[tau + 1]
                G /= gamma
            
            t += 1
    return V
