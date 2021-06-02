import numpy as np

class KArmedBandits(object):
    def __init__(self, variance=1.0):
        self.variance = variance
        self.reset()
    
    def draw(self, action):
        return np.random.normal(loc=self.q[action], scale=self.variance)

    def best_action(self):
        return np.argmax(self.q)
    
    def update(self):
        pass

    def reset(self):
        pass

class StationaryKArmedBandits(KArmedBandits):
    def __init__(self, k, mean=0.0, variance=1.0):
        self.k = k
        self.mean = mean
        super(StationaryKArmedBandits, self).__init__(variance)

    def reset(self):
        self.q = np.random.normal(loc=self.mean, scale=self.variance, size=self.k)

class MovingKArmedBandits(StationaryKArmedBandits):
    def __init__(self, k, mean=0.0, variance=1.0, moving_variance=0.01, same_start=True):
        self.moving_variance = moving_variance
        self.same_start = same_start
        super(MovingKArmedBandits, self).__init__(k, mean, variance)

    def update(self):
        self.q += np.random.normal(scale=self.moving_variance, size=self.q.shape)

    def reset(self):
        if self.same_start:
            self.q = np.ones(self.k) * np.random.normal(loc=self.mean, scale=self.variance)
        else:
            super(MovingKArmedBandits, self).reset()
