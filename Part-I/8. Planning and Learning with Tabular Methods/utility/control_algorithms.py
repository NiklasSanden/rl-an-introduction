import numpy as np
from tqdm import tqdm

from .misc import *

import random
import math
from collections import defaultdict

def q_learning_update(env, gamma, alpha, Q, state, action, reward, next_state):
    '''
    If next_state is terminal it should be None
    '''
    if next_state:
        actions = env.get_actions(next_state)
        max_q_next_state = np.max(np.array([Q[(next_state, a)] for a in actions]))
        Q[(state, action)] += alpha * (reward + gamma * max_q_next_state - Q[(state, action)])
    else:
        Q[(state, action)] += alpha * (reward - Q[(state, action)])

def dyna_Q(env, agent, gamma, max_episodes, n, alpha=0.1, Q=defaultdict(lambda: 0), model=dict(), model_keys=[], log=True):
    for _ in tqdm(range(max_episodes), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state, Q)
            next_state, reward, terminal, _ = env.step(action)
            if terminal:
                next_state = None
                
            # direct rl
            q_learning_update(env, gamma, alpha, Q, state, action, reward, next_state)

            # update model
            if not ((state, action) in model):
                model_keys.append((state, action))
            model[(state, action)] = (reward, next_state)

            # planning
            for _ in range(n):
                s, a = random.choice(model_keys)
                r, s_ = model[(s, a)]
                q_learning_update(env, gamma, alpha, Q, s, a, r, s_)
            
            state = next_state

    return (Q, model, model_keys)

def dyna_Q_plus(env, agent, gamma, max_episodes, n, kappa, alpha=0.1, Q=defaultdict(lambda: 0), model=dict(), model_keys=[], 
                tau=dict(), time_step=0, log=True):
    '''
    As a footnote in the book describes, for every state-action pair not visited, the model should expect it to transition
    to itself with a reward of 0. 
    '''
    for _ in tqdm(range(max_episodes), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state, Q)
            next_state, reward, terminal, _ = env.step(action)
            if terminal:
                next_state = None
                
            # direct rl
            q_learning_update(env, gamma, alpha, Q, state, action, reward, next_state)

            # update model
            if not ((state, action) in model):
                model_keys.append((state, action))
            model[(state, action)] = (reward, next_state)

            tau[(state, action)] = time_step

            # planning
            for _ in range(n):
                s, a = random.choice(model_keys)
                r, s_ = model[(s, a)]
                q_learning_update(env, gamma, alpha, Q, s, a, r + kappa * math.sqrt(time_step - tau[(s, a)]), s_)
            

            state = next_state
            time_step += 1

    return (Q, model, model_keys, tau, time_step)

def dyna_Q_plus_greedy_exercise_8_4(env, gamma, max_episodes, n, kappa, alpha=0.1, Q=defaultdict(lambda: 0), model=dict(), model_keys=[], 
                                    tau=dict(), time_step=0, log=True):
    '''
    As a footnote in the book describes, for every state-action pair not visited, the model should expect it to transition
    to itself with a reward of 0. This is especially important here since this implementation is ALWAYS completely greedy.
    '''
    for _ in tqdm(range(max_episodes), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            # action selection
            actions = env.get_actions(state)
            best_index = argmax(np.array([Q[(state, a)] + kappa * math.sqrt(time_step - tau[(state, a)]) if (state, a) in tau else Q[(state, a)] for a in actions]))
            action = actions[best_index]

            next_state, reward, terminal, _ = env.step(action)
            if terminal:
                next_state = None
                
            # direct rl
            q_learning_update(env, gamma, alpha, Q, state, action, reward, next_state)

            # update model
            if not ((state, action) in model):
                model_keys.append((state, action))
            model[(state, action)] = (reward, next_state)

            tau[(state, action)] = time_step

            # planning
            for _ in range(n):
                s, a = random.choice(model_keys)
                r, s_ = model[(s, a)]
                q_learning_update(env, gamma, alpha, Q, s, a, r, s_)
            

            state = next_state
            time_step += 1

    return (Q, model, model_keys, tau, time_step)
