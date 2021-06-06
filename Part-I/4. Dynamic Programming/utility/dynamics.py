import numpy as np
from tqdm import tqdm
import pickle

class ActionNode(object):
    def __init__(self, future_states=[], rewards=[], probabilities=[]):
        assert len(future_states) == len(rewards) == len(probabilities), 'Lists given to an ActionNode were of different lengths'
        
        self.transitions = []
        for (s_, r, p) in zip(future_states, rewards, probabilities):
            self.add_transition(s_, r, p)

    def add_transition(self, s_, r, p):
        self.transitions.append((s_, r, p))

    def get_transitions(self):
        return self.transitions

class StateNode(object):
    def __init__(self, actions, action_nodes=[]):
        assert (not action_nodes) or (len(action_nodes) == len(action_nodes)), 'action_nodes was not empty and its length was not equal to the length of actions'
        self.actions = actions
        self.transitions = {}
        for (idx, action) in enumerate(actions):
            self.transitions[action] = ActionNode() if not action_nodes else action_nodes[idx]
    
    def add_action(self, action, action_node):
        self.actions.append(action)
        self.transitions[action] = action_node

    def get_transitions(self, action):
        return self.transitions[action].get_transitions()
    
    def get_actions(self):
        return self.actions

class DynamicsFunction(object):
    def __init__(self, states):
        self.states = states
    
    def get_transitions(self, s, a):
        return self.states[s].get_transitions(a)
    
    def get_states(self):
        return self.states.keys()

    def get_state(self, s):
        return self.states[s]

    def get_actions(self, s):
        return self.states[s].get_actions()
    
class DynamicsFunctionPickle(DynamicsFunction):
    '''
    If the dynamics function is too large to keep in memory, this allows you to save each StateNode to disk and read it when needed.
    '''
    def __init__(self, states):
        super(DynamicsFunctionPickle, self).__init__(states)

    def get_transitions(self, s, a):
        return self.get_state(s).get_transitions(a)
    
    def get_state(self, s):
        with open('P/' + str(s), 'rb') as f:
            p = pickle.load(f)
            return p

    def get_actions(self, s):
        return self.get_state(s).get_actions()

    def save_state(self, name, s):
        with open('P/' + name, 'wb') as f:
            pickle.dump(s, f, protocol=-1)

class GridworldNxM(DynamicsFunction):
    def __init__(self, N=4, M=4):
        self.N = N
        self.M = M

        states = {}
        for n in range(N):
            for m in range(M):
                action_nodes = [ActionNode([s_], [-1], [1.0]) for s_ in self._get_adj_states(n, m)]
                states[(n, m)] = StateNode([0, 1, 2, 3], action_nodes)
        
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

class JacksCarRental(DynamicsFunction):
    '''
    cut_random_loop is necessary to make sure the P function fits in memory. A previous version inherited from DynamicsFunctionPickle to keep it on disk,
    but the amount of computation time to perform just a single iteration inside v_evaluation was too much for the entire P function.
    '''
    def __init__(self, max_a=20, max_b=20, max_transfer=5, lambda_customer_a=3.0, lambda_customer_b=4.0, 
                       lambda_return_a=3.0, lambda_return_b=2.0, pay=10.0, cost=2.0, extended=False, cut_random_loop=10):

        self.memo = {}

        states = {}
        for state_a in tqdm(range(max_a + 1)):
            for state_b in tqdm(range(max_b + 1), leave=False):
                action_nodes = []
                actions = []

                for action in range(-max_transfer, max_transfer + 1):
                    next_states = []
                    rewards = []
                    probabilites = []

                    new_state_a = state_a - action
                    new_state_b = state_b + action

                    if new_state_a <= max_a and new_state_a >= 0 and new_state_b <= max_b and new_state_b >= 0:

                        total_customer_a_prob = 0.0
                        customer_a_iter = min(cut_random_loop, new_state_a)
                        for customer_a in range(customer_a_iter + 1):
                            customer_a_prob = self.poisson(lambda_customer_a, customer_a) if customer_a < customer_a_iter else 1.0 - total_customer_a_prob
                            total_customer_a_prob += customer_a_prob

                            total_customer_b_prob = 0.0
                            customer_b_iter = min(cut_random_loop, new_state_b)
                            for customer_b in range(customer_b_iter + 1):
                                customer_b_prob = self.poisson(lambda_customer_b, customer_b) if customer_b < customer_b_iter else 1.0 - total_customer_b_prob
                                total_customer_b_prob += customer_b_prob

                                total_return_a_prob = 0.0
                                return_a_iter = min(cut_random_loop, max_a - (new_state_a - customer_a))
                                for return_a in range(return_a_iter + 1):
                                    return_a_prob = self.poisson(lambda_return_a, return_a) if return_a < return_a_iter else 1.0 - total_return_a_prob
                                    total_return_a_prob += return_a_prob

                                    total_return_b_prob = 0.0
                                    return_b_iter = min(cut_random_loop, max_b - (new_state_b - customer_b))
                                    for return_b in range(return_b_iter + 1):
                                        return_b_prob = self.poisson(lambda_return_b, return_b) if return_b < return_b_iter else 1.0 - total_return_b_prob
                                        total_return_b_prob += return_b_prob

                                        if extended and action > 0: # Jack's employee helps with one
                                            reward = -cost * (action - 1)
                                        else:
                                            reward = -cost * abs(action)
                                        reward += (customer_a + customer_b) * pay

                                        if extended: # Extra parking slots
                                            if new_state_a > 10:
                                                reward -= 4
                                            if new_state_b > 10:
                                                reward -= 4

                                        next_state_a = new_state_a - customer_a + return_a
                                        next_state_b = new_state_b - customer_b + return_b
                                        
                                        next_states.append((next_state_a, next_state_b))
                                        rewards.append(reward)
                                        probabilites.append(customer_a_prob * customer_b_prob * return_a_prob * return_b_prob)

                        actions.append(action)
                        action_nodes.append(ActionNode(next_states, rewards, probabilites))

                states[(state_a, state_b)] = StateNode(actions, action_nodes)

        super(JacksCarRental, self).__init__(states)

    def poisson(self, lambda_, n):
        if not ((lambda_, n) in self.memo):
            self.memo[(lambda_, n)] = (lambda_ ** n) * np.math.exp(-lambda_) / np.math.factorial(n)
        return self.memo[(lambda_, n)]


class GamblersProblem(DynamicsFunction):
    '''
    This implementation does not allow action 0 in states that aren't terminal. The figure in the book
    will break ties for the policy by (seemingly) choosing the lowest action/bet that isn't 0, so this 
    implementation detail makes it easier to get the same result.
    '''
    def __init__(self, p=0.4, goal=100):
        states = {}
        for state in range(1, goal):
            action_nodes = [ActionNode([state + action, state - action], 
                                       [1.0 if state + action == goal else 0.0, 0.0], 
                                       [p, 1.0 - p])
                                       for action in range(1, min(state, goal - state) + 1)]
            states[state] = StateNode([action for action in range(1, len(action_nodes) + 1)], action_nodes)
        for state in [0, goal]:
            action_nodes = [ActionNode([state], [0.0], [1.0])]
            states[state] = StateNode([0], action_nodes)

        super(GamblersProblem, self).__init__(states)
