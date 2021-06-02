import numpy as np

class QApproximator(object):
    def __init__(self, k):
        self.k = k
        self.reset()
    
    def get(self):
        return self.Q

    def update_action(self, action, value):
        pass

    def reset(self):
        pass

class QSampleAvg(QApproximator):
    def __init__(self, k):
        super(QSampleAvg, self).__init__(k)

    def update_action(self, action, value):
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (value - self.Q[action])
    
    def reset(self):
        self.N = np.zeros(self.k, dtype=int)
        self.Q = np.zeros(self.k)

class QConstant(QApproximator):
    def __init__(self, k, alpha):
        self.alpha = alpha
        super(QConstant, self).__init__(k)
    
    def update_action(self, action, value):
        self.Q[action] += self.alpha * (value - self.Q[action])

    def reset(self):
        self.Q = np.zeros(self.k)

class QConstantOptimistic(QConstant):
    def __init__(self, k, alpha, start_values):
        self.start_values = start_values
        super(QConstantOptimistic, self).__init__(k, alpha)
    
    def reset(self):
        self.Q = np.ones(self.k) * self.start_values

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
class GradientBandit(QApproximator): # Note that this approximates H instead of Q.
    def __init__(self, k, alpha, use_baseline=True):
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.baseline = Baseline()
        super(GradientBandit, self).__init__(k)

    def get(self):
        e_x = np.exp(self.H - np.max(self.H))
        return e_x / e_x.sum()

    def update_action(self, action, value):
        pi = self.get()
        if self.use_baseline:
            self.baseline.update(value)
            diff = self.baseline.get_diff(value)
        else:
            diff = value

        self.H -= self.alpha * diff * pi
        self.H[action] += self.alpha * diff

    def reset(self):
        self.H = np.zeros(self.k)
        self.baseline.reset()
