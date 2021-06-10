import numpy as np

class Environment(object):
    def __init__(self):
        pass

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

class SimplifiedBlackjack(Environment):
    def __init__(self):
        self.reset()

    def get_actions(self, state):
        return [0, 1]

    def step(self, action):
        '''
        action = 0 : hit
        action = 1 : stick
        '''
        if action == 0:
            self.player_sum, self.usable_ace = self._get_new_sum(self.player_sum, self.usable_ace)

            if self.player_sum > 21:
                return ((self.dealer_card, self.player_sum, self.usable_ace), -1, True, {})
            else:
                return ((self.dealer_card, self.player_sum, self.usable_ace), 0, False, {})
        elif action == 1: # dealer plays - behaviour is undefined if you stick after a terminal state
            usable_ace = self.dealer_card == 1
            dealer_sum = 11 if self.dealer_card == 1 else self.dealer_card
            while dealer_sum < 17:
                dealer_sum, usable_ace = self._get_new_sum(dealer_sum, usable_ace)
            
            if dealer_sum > 21 or dealer_sum < self.player_sum:
                reward = 1
            elif dealer_sum == self.player_sum:
                reward = 0
            else:
                reward = -1
            return ((self.dealer_card, self.player_sum, self.usable_ace), reward, True, {})
        else:
            Exception('Undefined action:', action)

    def reset(self):
        self.dealer_card = self._draw_card()
        self.usable_ace = False
        self.player_sum = 0
        while self.player_sum < 12:
            self.step(0)

        return (self.dealer_card, self.player_sum, self.usable_ace)

    def exploring_starts_reset(self):
        '''
        This is used by exploring starts methods to give non-zero probability to starting in any state-action pair 
        (in this case equal probability).
        '''
        self.dealer_card = np.random.choice(np.arange(1, 11))
        self.usable_ace = np.random.rand() < 0.5
        self.player_sum = np.random.choice(np.arange(12, 22))
        action = np.random.choice(np.arange(0, 2))

        return ((self.dealer_card, self.player_sum, self.usable_ace), action)

    def set_state(self, dealer_card, player_sum, usable_ace):
        self.dealer_card = dealer_card
        self.player_sum = player_sum
        self.usable_ace = usable_ace

    def _draw_card(self):
        return min(10, np.random.choice(np.arange(1, 14)))

    def _get_new_sum(self, sum, usable_ace):
        '''
        Hits and calculates the new sum. Also returns if you have a usable ace left.
        '''
        new_card = self._draw_card()
        sum += new_card
        if new_card == 1 and sum + 10 <= 21:
            sum += 10
            usable_ace = True
        if sum > 21 and usable_ace:
            sum -= 10
            usable_ace = False
        return (sum, usable_ace)

class BlackjackSingleStartState(SimplifiedBlackjack):
    def __init__(self, dealer_start, player_start, usable_ace_start):
        self.dealer_start = dealer_start
        self.player_start = player_start
        self.usable_ace_start = usable_ace_start
        super(BlackjackSingleStartState, self).__init__()

    def reset(self):
        self.set_state(self.dealer_start, self.player_start, self.usable_ace_start)
        return (self.dealer_card, self.player_sum, self.usable_ace)

class InfiniteVarianceLoop(Environment):
    def __init__(self):
        self.reset()

    def get_actions(self, state):
        return [0, 1]

    def step(self, action):
        '''
        action = 0 : left
        action = 1 : right
        '''
        if action == 0:
            if np.random.rand() < 0.9:
                return (self.state, 0.0, False, {})
            else:
                return (self.terminal_state, 1.0, True, {})
        elif action == 1:
            return (self.terminal_state, 0.0, True, {})
        else:
            Exception('Undefined action:', action)

    def reset(self):
        self.state = 0
        self.terminal_state = 1
        return self.state

class Racetrack(Environment):
    def __init__(self, max_y_speed, max_x_speed):
        self.max_y_speed = max_y_speed
        self.max_x_speed = max_x_speed
        self.reset()

    def get_actions(self, state):
        return [(y, x) for y in range(-1, 2) for x in range(-1, 2)]

    def step(self, action):
        if np.random.rand() < 0.1:
            action = (0, 0)
        return self.step_after_noise(action)
    
    def step_after_noise(self, action):
        dy, dx = action
        self.vel_y = max(0, min(self.max_y_speed, self.vel_y + dy))
        self.vel_x = max(0, min(self.max_x_speed, self.vel_x + dx))
        self._advance_time()

        if self._is_on_finish_line():
            return ((self.y, self.x, self.vel_y, self.vel_x), 0.0, True, {})
        elif self._is_OOB():
            self.reset()
        return ((self.y, self.x, self.vel_y, self.vel_x), -1.0, False, {})

    def reset(self):
        self._construct_track()
        self.y, self.x = self.starting_line[np.random.choice(np.arange(len(self.starting_line)))]
        self.vel_y = 0
        self.vel_x = 0
        return (self.y, self.x, self.vel_y, self.vel_x)

    def _construct_track(self):
        NotImplementedError()

    def _advance_time(self):
        new_y = self.y - self.vel_y
        new_x = self.x + self.vel_x
        while new_y != self.y or new_x != self.x:
            dy = min(1, abs(new_y - self.y)) * np.sign(new_y - self.y)
            dx = min(1, abs(new_x - self.x)) * np.sign(new_x - self.x)
            self.y += dy
            self.x += dx

            if self._is_OOB() or self._is_on_finish_line():
                return

    def _is_OOB(self):
        if self.y >= self.track.shape[0] or self.y < 0 or self.x >= self.track.shape[1] or self.x < 0:
            return True
        return not self.track[self.y, self.x]
    
    def _is_on_finish_line(self):
        return (self.y, self.x) in self.finish_line

class Racetrack_1(Racetrack):
    def __init__(self, max_y_speed=5, max_x_speed=5):
        super(Racetrack_1, self).__init__(max_y_speed, max_x_speed)

    def _construct_track(self):
        self.track = np.zeros((32, 17), dtype=bool)
        self.track[0, 3:] = True
        self.track[1:3, 2:] = True
        self.track[3, 1:] = True
        self.track[4:6, :] = True
        self.track[6, :10] = True
        self.track[7:14, :9] = True
        self.track[14:22, 1:9] = True
        self.track[22:29, 2:9] = True
        self.track[29:, 3:9] = True
        
        self.starting_line = [(31, x) for x in range(3, 9)]
        self.finish_line = [(y, 16) for y in range(6)]
