class Environment(object):
    def get_actions(self, state):
        '''
        Returns the available actions for the state
        '''
        NotImplementedError()

    def step(self, action):
        '''
        Returns the observation of the next state, the reward, a terminal flag and a dictionary with debug information
        '''
        NotImplementedError()

    def render(self):
        pass

    def reset(self):
        '''
        Returns the initial observation of the starting state
        '''
        NotImplementedError()

class CumulativeRewardWrapper(Environment):
    def __init__(self, environment):
        self.environment = environment
        self.reset()

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def get_actions(self, state):
        return self.environment.get_actions(state)

    def step(self, action):
        next_state, reward, terminal, info = self.environment.step(action)
        self.cumulative_reward += reward
        return (next_state, reward, terminal, info)

    def render(self):
        self.environment.render()

    def reset(self):
        self.cumulative_reward = 0
        return self.environment.reset()

class CumulativeRewardPerTimestepWrapper(CumulativeRewardWrapper):
    def __init__(self, environment):
        self.environment = environment
        self.reset_rewards()

    def get_cumulative_rewards(self):
        return self.cumulative_rewards

    def step(self, action):
        next_state, reward, terminal, info = super().step(action)
        self.cumulative_rewards.append(self.get_cumulative_reward())
        return (next_state, reward, terminal, info)

    def reset(self):
        return self.environment.reset()

    def reset_rewards(self):
        self.cumulative_rewards = []
        super().reset()    

class DynaMaze(Environment):
    def __init__(self, rows=6, cols=9, start=(2, 0), end=(0, 8), blocks={(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)}):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end
        self.blocks = blocks
        self.reset()

    def get_actions(self, state):
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    def step(self, action):
        new_pos = (self.pos[0] + action[0], self.pos[1] + action[1])
        if not self._is_invalid_state(new_pos):
            self.pos = new_pos
        
        terminal = self.end == self.pos
        return (self.pos, 1 if terminal else 0, terminal, {})

    def reset(self):
        self.pos = self.start
        return self.pos

    def _is_invalid_state(self, state):
        return (state in self.blocks) or state[0] < 0 or state[0] >= self.rows or state[1] < 0 or state[1] >= self.cols

class ChangingMaze(DynaMaze):
    def __init__(self, rows, cols, start, end, blocks, changed_blocks, time_steps_until_change=1000):
        self.start_blocks = blocks
        self.changed_blocks = changed_blocks
        self.time_steps_until_change = time_steps_until_change
        self.time_steps_left = time_steps_until_change
        super().__init__(rows, cols, start, end, blocks)
    
    def step(self, action):
        self.time_steps_left -= 1
        if self.time_steps_left == 0:
            self.blocks = self.changed_blocks
        return super().step(action)

    def reset_change(self):
        self.blocks = self.start_blocks
        self.time_steps_left = self.time_steps_until_change
