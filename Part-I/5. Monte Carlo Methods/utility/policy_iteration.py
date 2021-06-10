import numpy as np
from tqdm import tqdm

from collections import defaultdict

def monte_carlo_v_prediction(env, agent, gamma, max_iterations, start_V=defaultdict(lambda: 0.0), start_N=defaultdict(lambda: 0)):
    '''
    Unlike the pseudo-code in 5.1, this is both every-visit MC and it uses running averages (as described in Ch. 2)
    in order to not keep track of all of the returns.
    The agent should be deterministic and callable with the state as parameter.
    '''
    V = start_V
    N = start_N
    
    for i in tqdm(range(max_iterations)):
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
            N[states[i]] += 1
            V[states[i]] += (1.0 / N[states[i]]) * (G - V[states[i]])

    return (V, N)

def monte_carlo_q_exploring_starts(env, agent, gamma, max_iterations, start_Q=defaultdict(lambda: 0.0), start_N=defaultdict(lambda: 0)):
    '''
    Unlike the pseudo-code in 5.3, this is both every-visit MC and it uses running averages (as described in Ch. 2, 
    and requested in Exercise 5.4) in order to not keep track of all of the returns.
    The agent should be callable but could be stochastic. Its parameters should be the state and the estimated Q-function.
    '''
    Q = start_Q
    N = start_N
    
    for i in tqdm(range(max_iterations)):
        s, a = env.exploring_starts_reset()
        states = [s]
        actions = [a]
        rewards = []
        terminal = False
        while not terminal:
            next_state, reward, terminal, _ = env.step(actions[-1])
            if not terminal:
                states.append(next_state)
                actions.append(agent(Q, states[-1]))
            rewards.append(reward)
        assert len(states) == len(actions) == len(rewards), 'The sequence of experience have different lengths'
        G = 0
        for i in reversed(range(len(actions))):
            G = gamma * G + rewards[i]
            N[(states[i], actions[i])] += 1
            Q[(states[i], actions[i])] += (1.0 / N[(states[i], actions[i])]) * (G - Q[(states[i], actions[i])])

    return (Q, start_N)

def monte_carlo_v_off_policy(env, target, behaviour, gamma, max_iterations, start_V=defaultdict(lambda: 0.0), start_C=defaultdict(lambda: 0.0), weighted=True):
    '''
    target and behaviour should be callable to sample an action and have a method (get_prob) that takes in the action and state and returns
    the probability of that action being picked in that state.
    '''
    V = start_V
    C = start_C

    for i in tqdm(range(max_iterations), disable=True):
        states = [env.reset()]
        actions = []
        rewards = []
        terminal = False
        while not terminal:
            actions.append(behaviour(states[-1]))
            next_state, reward, terminal, _ = env.step(actions[-1])
            if not terminal:
                states.append(next_state)
            rewards.append(reward)
        assert len(states) == len(actions) == len(rewards), 'The sequence of experience have different lengths'
        G = 0
        W = 1.0
        for i in reversed(range(len(actions))):
            G = gamma * G + rewards[i]
            W *= target.get_prob(actions[i], states[i]) / behaviour.get_prob(actions[i], states[i])
            if weighted:
                if W == 0:
                    break
                C[states[i]] += W
                V[states[i]] += (W / C[states[i]]) * (G - V[states[i]])
            else:
                C[states[i]] += 1
                V[states[i]] += (1.0 / C[states[i]]) * (W * G - V[states[i]])

    return (V, C)

def monte_carlo_q_off_policy_control(env, target, behaviour, gamma, max_iterations, target_Q=defaultdict(lambda: 0.0), target_C=defaultdict(lambda: 0.0),
                                     behaviour_Q=defaultdict(lambda: 0.0), behaviour_N=defaultdict(lambda: 0)):
    '''
    This supports the behaviour policy changing by using on-policy MC control to learn its Q function.
    The target and the behaviour policies should be callable with the state and return a single action. 
    The behaviour policy should also have a method get_prob that returns the probability of taking action a in state s.
    The target and behaviour policies should also have a method called update_Q that takes in the latest Q estimates.
    '''
    for i in tqdm(range(max_iterations), disable=False):
        states = [env.reset()]
        actions = []
        rewards = []
        terminal = False
        while not terminal:
            actions.append(behaviour(states[-1]))
            next_state, reward, terminal, _ = env.step(actions[-1])
            if not terminal:
                states.append(next_state)
            rewards.append(reward)
        assert len(states) == len(actions) == len(rewards), 'The sequence of experience have different lengths'
        G = 0
        W = 1.0
        for i in reversed(range(len(actions))):
            G = gamma * G + rewards[i]
            # Behaviour update
            behaviour_N[(states[i], actions[i])] += 1
            behaviour_Q[(states[i], actions[i])] += (1.0 / behaviour_N[(states[i], actions[i])]) * (G - behaviour_Q[(states[i], actions[i])])

            # Target update
            if W != 0:
                target_C[(states[i], actions[i])] += W
                target_Q[(states[i], actions[i])] += (W / target_C[(states[i], actions[i])]) * (G - target_Q[(states[i], actions[i])])
                
                target.update_Q(target_Q)
                if actions[i] != target(states[i]):
                    W = 0

                W *= 1.0 / behaviour.get_prob(actions[i], states[i])

        behaviour.update_Q(behaviour_Q.copy()) # use a copy to not mess with the probability that is multiplied with W

    return (target_Q, target_C, behaviour_Q, behaviour_N)

def monte_carlo_q_on_policy_control(env, agent, gamma, max_iterations, start_Q=defaultdict(lambda: 0.0), start_N=defaultdict(lambda: 0)):
    '''
    The agent should be callable with the state and return a single action.
    It should also have a method called update_Q that takes in the latest Q estimates.
    '''
    Q = start_Q
    N = start_N

    for i in tqdm(range(max_iterations), disable=False):
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
            N[(states[i], actions[i])] += 1
            Q[(states[i], actions[i])] += (1.0 / N[(states[i], actions[i])]) * (G - Q[(states[i], actions[i])])

            agent.update_Q(Q)

    return (Q, N)
