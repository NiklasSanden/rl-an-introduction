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
