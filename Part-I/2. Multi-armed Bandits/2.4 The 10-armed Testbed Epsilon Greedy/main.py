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

class KArmedBandits(object):
    def __init__(self, k, variance=1.0):
        self.k = k
        self.variance = variance
        self.reset()

    def draw(self, action):
        return np.random.normal(loc=self.q[action], scale=self.variance)

    def best_action(self):
        return np.argmax(self.q)

    def reset(self):
        self.q = np.random.normal(scale=self.variance, size=self.k)

# PARAMETERS
K = 10
VARIANCE = 1.0
NUM_RUNS = 2000
STEPS = 1000
EPSILONS = [0.0, 0.01, 0.1]
COLOURS = ['g', 'r', 'b']
LABELS = ['eps=0.0', 'eps=0.01', 'eps=0.1']

if __name__ == "__main__":
    agents = [EpsilonGreedy(K, i) for i in EPSILONS]
    bandits = KArmedBandits(K, VARIANCE)
    average_reward = np.zeros((NUM_RUNS, STEPS, len(agents)))
    best_action_precentage = np.zeros((NUM_RUNS, STEPS, len(agents)))
    for i in tqdm(range(NUM_RUNS)):
        for agent in agents:
            agent.reset()
        bandits.reset()
        best_action = bandits.best_action()

        for j in range(STEPS):
            for a in range(len(agents)):
                action = agents[a].select_action()
                reward = bandits.draw(action)
                agents[a].update_action(action, reward)

                average_reward[i, j, a] = reward
                best_action_precentage[i, j, a] = 100 if action == best_action else 0
    
    average_reward = np.average(average_reward, axis=0)
    best_action_precentage = np.average(best_action_precentage, axis=0)

    fig, axs = plt.subplots(2)
    for i in range(len(EPSILONS)):
        axs[0].plot(average_reward[:, i], color=COLOURS[i], label=LABELS[i])
        axs[1].plot(best_action_precentage[:, i], color=COLOURS[i], label=LABELS[i])
    axs[0].set(ylabel='Average Reward', xlabel='Steps')
    axs[1].set(ylabel='Optimal action %', xlabel='Steps')
    axs[0].legend()
    axs[1].legend()
    plt.show()
        