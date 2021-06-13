from tqdm import tqdm

from collections import defaultdict

def sarsa_on_policy_td_q(env, agent, gamma, max_iterations, alpha=0.1, start_Q=defaultdict(lambda: 0.0), log=True):
    '''
    The agent should be callable with the state and Q-function as parameters and return one action.
    '''
    Q = start_Q

    for i in tqdm(range(max_iterations), disable=(not log)):
        state = env.reset()
        action = agent(state, Q)
        terminal = False
        while not terminal:
            next_state, reward, terminal, _ = env.step(action)
            if terminal: # just in case the user gave a start_Q where Q(terminal, a) != 0
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
            else:
                next_action = agent(next_state, Q)
                Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])
                action = next_action
            state = next_state

    return Q
