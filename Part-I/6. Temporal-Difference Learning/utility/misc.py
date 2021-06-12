from tqdm import tqdm

import math

def RMS_error(V, num_states, true_values):
    error = 0
    for s in range(num_states):
        error += (V[s] - true_values[s]) ** 2
    error /= num_states
    return math.sqrt(error)

def gather_experience_and_returns(env, agent, gamma, max_episodes, start_returns={}, start_experience_tuples={}, log=True):
    '''
    This plays out a number of episodes and returns a dictionary with lists with all of the returns for each state encountered (used for batch MC).
    It also returns a dictionary with a list of all reward & next-state tuples experienced for each state encountered (used for batch TD(0)).
    '''
    returns = start_returns
    experience_tuples = start_experience_tuples
    for i in tqdm(range(max_episodes), disable=(not log)):
        states = [env.reset()]
        actions = []
        rewards = []
        terminal = False
        while not terminal:
            actions.append(agent(states[-1]))
            next_state, reward, terminal, _ = env.step(actions[-1])
            states.append(next_state)
            rewards.append(reward)

            # experience tuples
            if states[-2] in experience_tuples:
                experience_tuples[states[-2]].append((rewards[-1], next_state))
            else:
                experience_tuples[states[-2]] = [(rewards[-1], next_state)]

        # Note that states contains the terminal state and will therefore have one more element than the other lists,
        # however, the terminal state won't be indexed in the loop below.
        # returns
        G = 0
        for i in reversed(range(len(actions))):
            G = gamma * G + rewards[i]
            if states[i] in returns:
                returns[states[i]].append(G)
            else:
                returns[states[i]] = [G]

    return (returns, experience_tuples)
