from tqdm import tqdm

from .misc import *

def REINFORCE(env, agent, gamma, max_episodes, alpha, log=True):
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
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
        
        Gs = [0] * (len(states) + 1)
        for i in reversed(range(len(states))):
            Gs[i] = rewards[i] + gamma * Gs[i + 1]

        gamma_iterator = 1
        for i in range(len(states)):
            change = gamma_iterator * Gs[i] * agent.get_ln_gradients(states[i], actions[i])
            agent.update_weights(alpha, change)
            gamma_iterator *= gamma

def REINFORCE_with_baseline(env, agent, gamma, max_episodes, alpha_theta, alpha_w, V, log=True):
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
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
        
        Gs = [0] * (len(states) + 1)
        for i in reversed(range(len(states))):
            Gs[i] = rewards[i] + gamma * Gs[i + 1]

        gamma_iterator = 1
        for i in range(len(states)):
            delta = Gs[i] - V(states[i])
            v_change = delta * V.get_gradients(states[i])
            pi_change = gamma_iterator * delta * agent.get_ln_gradients(states[i], actions[i])
            V.update_weights(alpha_w, v_change)
            agent.update_weights(alpha_theta, pi_change)
            gamma_iterator *= gamma
