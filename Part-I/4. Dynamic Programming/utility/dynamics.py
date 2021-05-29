class ActionNode(object):
    def __init__(self, future_states=[], rewards=[], probabilities=[]):
        assert len(future_states) == len(rewards) == len(probabilities), "Lists given to an ActionNode were of different lengths"
        
        self.transitions = []
        for (s_, r, p) in zip(future_states, rewards, probabilities):
            self.add_transition(s_, r, p)

    def add_transition(self, s_, r, p):
        self.transitions.append((s_, r, p))

    def get_transitions(self):
        return self.transitions

class StateNode(object):
    def __init__(self, actions, action_nodes=[]):
        assert (not action_nodes) or (len(action_nodes) == len(action_nodes)), "action_nodes was not empty and its length was not equal to the length of actions"
        self.transitions = {}
        for (idx, action) in enumerate(actions):
            self.transitions[action] = ActionNode() if not action_nodes else action_nodes[idx]
    
    def add_action(self, action, action_node):
        self.transitions[action] = action_node

    def get_transitions(self, action):
        return self.transitions[action].get_transitions()

class DynamicsFunction(object):
    def __init__(self, states):
        self.states = states
    
    def get_transitions(self, s, a):
        return self.states[s].get_transitions(a)
    
    def get_states(self):
        return self.states

    def get_actions(self):
        pass

class GridworldNxM(DynamicsFunction):
    def __init__(self, N, M):
        self.N = N
        self.M = M

        states = {}
        for n in range(N):
            for m in range(M):
                action_nodes = [ActionNode([s_], [-1], [1.0]) for s_ in self._get_adj_states(n, m)]
                states[(n, m)] = StateNode(self.get_actions(), action_nodes)
        
        super(GridworldNxM, self).__init__(states)
    
    def _get_adj_states(self, n, m):
        return [
            self._OOB_check((n, m), n + 1, m),
            self._OOB_check((n, m), n - 1, m),
            self._OOB_check((n, m), n, m + 1),
            self._OOB_check((n, m), n, m - 1)
        ]
    
    def _OOB_check(self, default, n, m):
        return default if n < 0 or n >= self.N or m < 0 or m >= self.M else (n, m)

    def get_actions(self):
        return [0, 1, 2, 3]
