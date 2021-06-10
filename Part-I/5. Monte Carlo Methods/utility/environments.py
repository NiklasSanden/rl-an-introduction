import numpy as np

class Environment(object):
    def __init__(self):
        pass

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
