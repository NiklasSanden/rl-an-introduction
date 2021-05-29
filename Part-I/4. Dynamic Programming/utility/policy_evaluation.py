from collections import defaultdict

def VEvaluation(PI, dynamics, terminals, max_delta, gamma):
    V = defaultdict(lambda: 0.0)
    actions = dynamics.get_actions()
    
    while True:
        delta = 0.0
        for state in dynamics.get_states():
            if state in terminals:
                continue

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
        
        if delta < max_delta:
            break
    
    return V
