from numpy.lib.twodim_base import diag
from tqdm import tqdm

from collections import defaultdict

def monte_carlo_v_prediction(env, agent, gamma, max_iterations, alpha=0.1, start_V=defaultdict(lambda: 0.0), log=True):
    '''
    Every-visit MC with constant step size parameter alpha.
    The agent should be callable with the state as parameter and return one action.
    '''
    V = start_V
    
    for i in tqdm(range(max_iterations), disable=(not log)):
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
            V[states[i]] += alpha * (G - V[states[i]])

    return V

def TD0_v_prediction(env, agent, gamma, max_iterations, alpha=0.1, start_V=defaultdict(lambda: 0.0), terminals=set(), log=True):
    '''
    The agent should be callable with the state as parameter and return one action.
    Terminals is a set of terminal states. Use this if the default value of start_V isn't 0.0. This will set
    the value of the terminal states to 0 for you.
    '''
    V = start_V
    for state in terminals:
        V[state] = 0.0
    
    for i in tqdm(range(max_iterations), disable=(not log)):
        state = env.reset()
        terminal = False
        while not terminal:
            action = agent(state)
            next_state, reward, terminal, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

    return V
