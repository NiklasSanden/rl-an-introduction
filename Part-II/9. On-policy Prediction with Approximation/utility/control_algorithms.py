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
