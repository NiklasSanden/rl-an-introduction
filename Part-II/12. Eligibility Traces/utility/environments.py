from .misc import *

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

class MaxStepsWrapper(object):
    def __init__(self, env, max_steps):
        self.env = env
        self.max_steps = max_steps
        self.reset_steps()
    
    def get_actions(self, state):
        return self.env.get_actions(state)

    def step(self, action):
        s_, r, t, info = self.env.step(action)
        self.steps += 1
        if self.steps > self.max_steps:
            t = True
        return (s_, r, t, info)

    def render(self):
        self.env.render()

    def reset(self):
        self.reset_steps()
        return self.env.reset()
    
    def reset_steps(self):
        self.steps = 0

class RandomWalk(Environment):
    def __init__(self, num_states=19):
        '''
        num_states is excluding the two terminal states on the left and right
        '''
        assert num_states >= 1 and (num_states % 2) == 1, str(num_states) + ' is not a valid num_states'
        self.num_states = num_states
        self.reset()

    def get_actions(self, state):
        return [-1, 1]
    
    def step(self, action):
        self.state += action
        terminal = False
        reward = 0.0
        if self.state < 0:
            terminal = True
            reward = -1.0
        elif self.state >= self.num_states:
            terminal = True
            reward = 1.0
        return (self.state, reward, terminal, {})

    def reset(self):
        self.state = self.num_states // 2
        return self.state

class MountainCar(Environment):
    def __init__(self, min_x=-1.2, max_x=0.5, min_v=-0.07, max_v=0.07, speed=0.001, gravity=0.0025, freq=3, start_x_min=-0.6, 
                 start_x_max=-0.4, start_v_min=0.0, start_v_max=0.0):
        self.min_x = min_x
        self.max_x = max_x
        self.min_v = min_v
        self.max_v = max_v
        self.speed = speed
        self.gravity = gravity
        self.freq = freq
        self.start_x_min = start_x_min
        self.start_x_max = start_x_max
        self.start_v_min = start_v_min
        self.start_v_max = start_v_max
        self.reset()

    def get_actions(self, state):
        return [-1, 0, 1]
    
    def step(self, action):
        unclipped_pos = self.pos + self.v
        self.pos = clip(unclipped_pos, self.min_x, self.max_x)
        self.v = clip(self.v + self.speed * action - self.gravity * math.cos(self.freq * self.pos), self.min_v, self.max_v)
        
        terminal = unclipped_pos > self.max_x
        if unclipped_pos < self.min_x:
            self.v = 0.0
        
        return ((self.pos, self.v), -1, terminal, {})

    def reset(self):
        r = np.random.rand(2)
        self.pos = r[0] * (self.start_x_max - self.start_x_min) + self.start_x_min
        self.v = r[1] * (self.start_v_max - self.start_v_min) + self.start_v_min
        return (self.pos, self.v)
