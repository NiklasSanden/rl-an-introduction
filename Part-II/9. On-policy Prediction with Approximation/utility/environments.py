import numpy as np

class Environment(object):
    def get_actions(self, state):
        '''
        Returns the available actions for the state
        '''
        raise NotImplementedError()

    def step(self, action):
        '''
        Returns the observation of the next state, the reward, a terminal flag and a dictionary with debug information
        '''
        raise NotImplementedError()

    def render(self):
        pass

    def reset(self):
        '''
        Returns the initial observation of the starting state
        '''
        raise NotImplementedError()

class RandomWalk(Environment):
    def __init__(self, num_states=1000, step_size=100):
        '''
        num_states is excluding the two terminal states on the left and right
        '''
        assert num_states >= 1, str(num_states) + ' is not a valid num_states'
        self.num_states = num_states
        self.step_size = step_size
        self.reset()

    def get_actions(self, state):
        return [-x for x in range(1, self.step_size + 1)] + [x for x in range(1, self.step_size + 1)]
    
    def step(self, action):
        self.state += action
        self.state = max(min(self.state, self.num_states + 1), 0)
        reward = 0.0
        terminal = False
        if self.state <= 0:
            terminal = True
            reward = -1.0
        elif self.state > self.num_states:
            terminal = True
            reward = 1.0
        return (self.state, reward, terminal, {})

    def reset(self):
        self.state = self.num_states // 2
        return self.state

    def get_true_value(self, gamma, max_delta, log=True):
        '''
        Returns the true v function for the uniform agent.
        '''
        V = np.zeros(self.num_states)
        while True:
            delta = 0.0
            for state in range(self.num_states):
                actions = self.get_actions(state)
                old_est = V[state]
                sum = 0.0
                for action in actions:
                    s_ = state + action
                    value_s_ = V[s_] if s_ < self.num_states and s_ >= 0 else 0
                    if s_ >= self.num_states:
                        r = 1
                    elif s_ < 0:
                        r = -1
                    else:
                        r = 0
                    sum += 1 / len(actions) * (r + gamma * value_s_)
                V[state] = sum

                delta = max(delta, abs(old_est - V[state]))
            
            if log:
                print('policy evaluation error:', delta)
            if delta <= max_delta:
                break
        
        return V
