import numpy as np

class Agent(object):
    def __init__(self, k, action_selector, q_approximator):
        self.action_selector = action_selector
        self.q_approximator = q_approximator

    def select_action(self):
        return self.action_selector(self.q_approximator.get())

    def update_action(self, action, value):
        self.q_approximator.update_action(action, value)

    def reset(self):
        self.action_selector.reset()
        self.q_approximator.reset()

class Baseline(object):
    def __init__(self):
        self.reset()

    def get_diff(self, R):
        if self.t == 0:
            return 0.0
        else:
            return R - self.R

    def update(self, R):
        self.t += 1
        self.R += (1.0 / self.t) * (R - self.R)

    def reset(self):
        self.R = 0.0
        self.t = 0

class GradientBandit(object):
    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha
        self.baseline = Baseline()
        self.reset()

    def get_probs(self):
        e_x = np.exp(self.H - np.max(self.H))
        return e_x / e_x.sum()

    def select_action(self):
        pi = self.get_probs()
        return np.random.choice(range(self.H.size), p=pi)

    def update_action(self, action, value):
        pi = self.get_probs()
        self.baseline.update(value)
        diff = self.baseline.get_diff(value)

        self.H -= self.alpha * diff * pi
        self.H[action] += self.alpha * diff

    def reset(self):
        self.H = np.zeros(self.k)
        self.baseline.reset()
