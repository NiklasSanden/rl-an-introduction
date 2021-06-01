from collections import defaultdict

def v_evaluation(PI, dynamics, terminals, max_delta, gamma, start_V=defaultdict(lambda: 0.0), log=True):
    V = start_V
    
    already_optimal = True
    while True:
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

        if delta < max_delta:
            break
        else:
            already_optimal = False
    
    return (V, already_optimal)

def get_greedy_PI_from_V(V, dynamics, gamma, log=True):
    PI = dict()

    for state in V: # won't contain terminal states
        actions = dynamics.get_actions(state)

        best_action = -1
        best_value = float('-inf')
        for action in actions:
            transitions = dynamics.get_transitions(state, action)
            sum = 0.0
            for (s_, r, p) in transitions:
                sum += p * (r + gamma * V[s_])
            if sum >= best_value:
                best_value = sum
                best_action = action

        PI[state] = best_action
    
    if log:
        print("Policy updated")

    return lambda s: { action : 1.0 if PI[s] == action else 0.0 for action in dynamics.get_actions(s) }
