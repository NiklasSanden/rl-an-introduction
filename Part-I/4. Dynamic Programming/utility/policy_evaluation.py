import numpy as np

from collections import defaultdict

from .misc import *

def v_evaluation(PI, dynamics, terminals, max_delta, gamma, start_V=defaultdict(lambda: 0.0), max_iterations=float('inf'), log=True):
    V = start_V
    
    already_optimal = True
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        delta = 0.0
        for state in dynamics.get_states():
            if state in terminals:
                continue

            actions = dynamics.get_actions(state)

            old_est = V[state]
            
            outer_sum = 0.0
            for action in actions:
                transitions = dynamics.get_transitions(state, action)
                inner_sum = 0.0
                for (s_, r, p) in transitions:
                    inner_sum += p * (r + gamma * V[s_])
                
                outer_sum += PI(state)[action] * inner_sum
            V[state] = outer_sum

            delta = max(delta, abs(old_est - V[state]))
        
        if log:
            print("Policy evaluation delta:", delta)

        if delta <= max_delta:
            break
        else:
            already_optimal = False
    
    return (V, already_optimal)

def get_greedy_PI_from_V(V, dynamics, gamma, break_ties_randomly=False, should_round=False, round_to_decimal=5, log=True):
    PI = dict()

    for state in V: # won't contain terminal states
        actions = dynamics.get_actions(state)

        action_values = np.zeros(len(actions))
        for idx, action in enumerate(actions):
            transitions = dynamics.get_transitions(state, action)
            for (s_, r, p) in transitions:
                action_values[idx] += p * (r + gamma * V[s_])
        
        if should_round:
            action_values = np.round(action_values, round_to_decimal)
        if break_ties_randomly:
            best_action_idx = argmax(action_values)
        else:
            best_action_idx = np.argmax(action_values)
        PI[state] = actions[best_action_idx]
    
    if log:
        print("Policy updated")

    return lambda s: { action : 1.0 if PI[s] == action else 0.0 for action in dynamics.get_actions(s) }

def v_value_iteration(dynamics, terminals, max_delta, gamma, start_V=defaultdict(lambda: 0.0), max_iterations=float('inf'), log=True):
    V = start_V
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        delta = 0.0
        for state in dynamics.get_states():
            if state in terminals:
                continue

            actions = dynamics.get_actions(state)

            old_est = V[state]
            
            best_value = float('-inf')
            for action in actions:
                transitions = dynamics.get_transitions(state, action)
                inner_sum = 0.0
                for (s_, r, p) in transitions:
                    inner_sum += p * (r + gamma * V[s_])
                
                best_value = max(best_value, inner_sum)
            V[state] = best_value

            delta = max(delta, abs(old_est - V[state]))
        
        if log:
            print("Value iteration delta:", delta)

        if delta <= max_delta:
            break
    
    return V
