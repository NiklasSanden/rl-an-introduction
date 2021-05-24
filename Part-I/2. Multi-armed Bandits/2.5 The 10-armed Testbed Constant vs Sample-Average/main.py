import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def argmax(array):
    """
    The numpy argmax function breaks ties by choosing the first occurence. This implementation breaks ties uniformly at random.
    """
    assert (len(array.shape) == 1), "argmax expects a 1d array"
    assert (array.size > 0),        "argmax expects non-empty array"

    indices = []
    best = float('-inf')
    for i in range(len(array)):
        if array[i] > best:
            best = array[i]
            indices = [i]
        elif array[i] == best:
            indices.append(i)
    return np.random.choice(indices)

class Agent(object):
    def __init__(self, k):
        self.k = k
        self.reset()

    def select_action(self):
        pass

    def update_action(self, action, value):
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (value - self.Q[action])

    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k, dtype=int)

class EpsilonGreedy(Agent):
    def __init__(self, k, epsilon):
        super(EpsilonGreedy, self).__init__(k)
        self.epsilon = epsilon
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.k)
        else:
            return argmax(self.Q)

class ConstantEpsilonGreedy(EpsilonGreedy):
    def __init__(self, k, epsilon, alpha):
        super(ConstantEpsilonGreedy, self).__init__(k, epsilon)
        self.alpha = alpha
    
    def update_action(self, action, value):
        self.N[action] += 1
        self.Q[action] += self.alpha * (value - self.Q[action])

class MovingKArmedBandits(object):
    def __init__(self, k, variance=1.0, moving_variance=0.01):
        self.k = k
        self.variance = variance
        self.moving_variance = moving_variance
        self.reset()

    def draw(self, action):
        return np.random.normal(loc=self.q[action], scale=self.variance)
    
    def move(self):
        self.q += np.random.normal(scale=self.moving_variance, size=self.q.shape)

    def best_action(self):
        return np.argmax(self.q)

    def reset(self):
        self.q = np.ones(self.k) * np.random.normal(scale=self.variance)

# PARAMETERS
K = 10
VARIANCE = 1.0
MOVING_VARIANCE = 0.01
NUM_RUNS = 2000
STEPS = 10000
AGENTS = [ConstantEpsilonGreedy(K, 0.1, 0.1), EpsilonGreedy(K, 0.1)]
COLOURS = ['r', 'b']
LABELS = ['Constant', 'Sample Avg']

if __name__ == "__main__":
    agents = AGENTS
    bandits = MovingKArmedBandits(K, VARIANCE, MOVING_VARIANCE)
    average_reward = np.zeros((NUM_RUNS, STEPS, len(agents)))
    best_action_precentage = np.zeros((NUM_RUNS, STEPS, len(agents)))
    for i in tqdm(range(NUM_RUNS)):
        for agent in agents:
            agent.reset()
        bandits.reset()

        for j in range(STEPS):
            best_action = bandits.best_action()
            for a in range(len(agents)):
                action = agents[a].select_action()
                reward = bandits.draw(action)
                agents[a].update_action(action, reward)

                average_reward[i, j, a] = reward
                best_action_precentage[i, j, a] = 100 if action == best_action else 0
            bandits.move()
    
    average_reward = np.average(average_reward, axis=0)
    best_action_precentage = np.average(best_action_precentage, axis=0)

    fig, axs = plt.subplots(2)
    for i in range(len(AGENTS)):
        axs[0].plot(average_reward[:, i], color=COLOURS[i], label=LABELS[i])
        axs[1].plot(best_action_precentage[:, i], color=COLOURS[i], label=LABELS[i])
    axs[0].set(ylabel='Average Reward', xlabel='Steps')
    axs[1].set(ylabel='Optimal action %', xlabel='Steps')
    axs[0].legend()
    axs[1].legend()
    plt.show()
        