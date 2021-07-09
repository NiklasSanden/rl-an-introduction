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
