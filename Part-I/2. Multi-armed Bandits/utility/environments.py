import numpy as np

class KArmedBandits(object):
    def __init__(self, k, variance):
        self.k = k
        self.variance = variance
        self.reset()
    
    def draw(self, action):
        return np.random.normal(loc=self.q[action], scale=self.variance)

    def best_action(self):
        return np.argmax(self.q)

    def reset(self):
        pass

class StationaryKArmedBandits(KArmedBandits):
    def __init__(self, k, variance):
        super(StationaryKArmedBandits, self).__init__(k, variance)

    def reset(self):
        self.q = np.random.normal(scale=self.variance, size=self.k)

class MovingKArmedBandits(KArmedBandits):
    def __init__(self, k, variance, moving_variance):
        self.moving_variance = moving_variance
        super(MovingKArmedBandits, self).__init__(k, variance)

    def move(self):
        self.q += np.random.normal(scale=self.moving_variance, size=self.q.shape)

    def reset(self):
        self.q = np.ones(self.k) * np.random.normal(scale=self.variance)
