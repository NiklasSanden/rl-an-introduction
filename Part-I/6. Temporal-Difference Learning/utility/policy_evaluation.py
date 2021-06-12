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

def batch_TD0(experience_tuples, gamma, max_delta=0.0001, start_V=defaultdict(lambda: 0.0), log=True):
    '''
    This does not use an alpha. The perfect alpha for each batch is just 1/b where b is the batch size for that state.
    You can empirically check this by replacing 1.0 / len(experience) below with either a fixed value or changing the 1.0 to something
    higher or lower. If it is higher, the estimates will most likely diverge, and if it is lower, they take longer to converge.
    Using 1/b makes each update equivalent to a DP policy evaluation update but with the dynamics and policy distributions
    being the maximum-likelihood versions given all of the samples we have.
    '''
    V = start_V

    counter = 0
    while True:
        delta = 0
        counter += 1
        old_V = V # set this to V.copy() if you don't want it to be in place
        for state in experience_tuples:
            increments = 0.0
            experience = experience_tuples[state]
            for R, next_state in experience:
                increments += 1.0 / len(experience) * (R + gamma * old_V[next_state] - old_V[state]) # - old_V[state] is not necessary if ...
            V[state] = V[state] + increments                                                         # ... this is set to = increments. They cancel out.
            delta = max(delta, increments)                                                           # I kept it like this since it is closer to the theory in the book.
        if log:
            print('From batch_TD0 - iteration =', counter, '- delta =', delta)
        if delta < max_delta:
            break
    
    return V

def batch_monte_carlo(returns, start_V=defaultdict(lambda: 0.0)):
    '''
    This does not use an alpha or max_delta. Instead of alpha it uses 1.0 / len(Gs) (see below) which immediately
    moves the estimates to the average of all of the returns in one sweep - no need for a loop.
    '''
    V = start_V

    for state in returns:
        increments = 0.0
        Gs = returns[state]
        for G in Gs:
            increments += 1.0 / len(Gs) * (G - V[state]) # this could be just 1.0 / len(Gs) * G if ...
        V[state] = V[state] + increments                 # ... this was just = increments. It is equivalent.
                                                         # I kept it like this since it is closer to the theory in the book.
    return V
