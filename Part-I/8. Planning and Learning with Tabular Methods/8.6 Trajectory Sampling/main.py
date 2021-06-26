import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utility.environments import *
from utility.misc import *
from utility.control_algorithms import *
from utility.agents import *

# PARAMETERS
SHOULD_COMPUTE_V = True
NUM_EPISODES_TO_ESTIMATE_V = 500
NUM_RUNS = [100, 15]
B_VALUES = [1, 3, 10]
NUM_STATES = [1000, 10000]
NUM_UPDATES = [20000, 200000]
EPSILON = 0.1
MAX_DELTA = 0.1
GAMMA = 1.0
NUM_TICKS = 20
COLOURS = ['green', 'red', 'blue']
LINESTYLES = ['solid', 'dashed']

def compute_V(Q, env, V=defaultdict(lambda: 0.0)):
    delta = float('inf')
    while delta > MAX_DELTA:
        delta = 0.0
        for state in range(env.num_states):
            actions = env.get_actions(state)
            action = actions[np.argmax(np.array([Q[state, a] for a in actions]))]

            old_est = V[state]
            
            sum = 0.0
            next_states, rewards, terminal_odds = env.get_transitions(state, action)
            for s_, r in zip(next_states, rewards):
                sum += r + (1.0 - terminal_odds) * GAMMA * V[s_]
            sum /= len(next_states)
                
            V[state] = sum
            delta = max(delta, abs(old_est - V[state]))
    return V

def estimate_V(Q, env_to_copy):
    env = TrajectorySamplingEnvironment(None, None, other_env=env_to_copy)
    returns = np.zeros(NUM_EPISODES_TO_ESTIMATE_V)
    for episode in range(NUM_EPISODES_TO_ESTIMATE_V):
        state = env.reset()
        terminal = False
        while not terminal:
            actions = env.get_actions(state)
            action = actions[argmax(np.array([Q[(state, a)] for a in actions]))]
            state, reward, terminal, _ = env.step(action)
            returns[episode] += reward
    return np.average(returns)

def plot_uniform(ax, env, b, num_updates, colour, linestyle, num_runs):
    tick_distance = num_updates // NUM_TICKS
    state_value = np.zeros((num_runs, NUM_TICKS + 1))
    for run in tqdm(range(num_runs), leave=False):
        V = defaultdict(lambda: 0)
        start_state = env.reset()
        Q = defaultdict(lambda: 0)
        for i in tqdm(range(num_updates), leave=False, disable=True):
            s, a = env.get_random_state_action_pair()
            next_states, rewards, terminal_odds = env.get_transitions(s, a)
            assert len(next_states) == b, 'the number of next_states should be b'
            sum = 0.0
            for s_, r in zip(next_states, rewards):
                sum += r + (1.0 - terminal_odds) * GAMMA * np.max([Q[(s_, a_)] for a_ in env.get_actions(s_)])
            Q[(s, a)] = sum / b

            # store V(s_0)
            if i % tick_distance == tick_distance - 1:
                if SHOULD_COMPUTE_V:
                    V = compute_V(Q, env, V)
                    value = V[start_state]
                else:
                    value = estimate_V(Q, env)
                state_value[run, i // tick_distance + 1] = value
    state_value = np.average(state_value, axis=0)
    ax.plot(np.arange(0, NUM_TICKS + 1) * tick_distance, state_value, label='uniform b=' + str(b), color=colour, linestyle=linestyle)

def plot_on_policy(ax, env, b, num_updates, colour, linestyle, num_runs):
    tick_distance = num_updates // NUM_TICKS
    state_value = np.zeros((num_runs, NUM_TICKS + 1))
    for run in tqdm(range(num_runs), leave=False):
        V = defaultdict(lambda: 0)
        start_state = env.reset()
        agent = EpsilonGreedyAgent(env, epsilon=EPSILON)
        Q = defaultdict(lambda: 0)
        terminal = True
        for i in tqdm(range(num_updates), leave=False, disable=True):
            if terminal:
                terminal = False
                state = env.reset()
            action = agent(state, Q)
            next_state, _, terminal, _ = env.step(action)
            
            next_states, rewards, terminal_odds = env.get_transitions(state, action)
            assert len(next_states) == b, 'the number of next_states should be b'
            sum = 0.0
            for s_, r in zip(next_states, rewards):
                sum += r + (1.0 - terminal_odds) * GAMMA * np.max([Q[(s_, a_)] for a_ in env.get_actions(s_)])
            Q[(state, action)] = sum / b

            state = next_state

            # store V(s_0)
            if i % tick_distance == tick_distance - 1:
                if SHOULD_COMPUTE_V:
                    V = compute_V(Q, env, V)
                    value = V[start_state]
                else:
                    value = estimate_V(Q, env)
                state_value[run, i // tick_distance + 1] = value
    state_value = np.average(state_value, axis=0)
    ax.plot(np.arange(0, NUM_TICKS + 1) * tick_distance, state_value, label='on-policy b=' + str(b), color=colour, linestyle=linestyle)

if __name__ == '__main__':
    fig = plt.figure()

    for n in tqdm(range(len(NUM_STATES))):
        ax = fig.add_subplot(2, 1, 1 + n)
        for i in tqdm(range(len(B_VALUES)), leave=False):
            env = TrajectorySamplingEnvironment(NUM_STATES[n], B_VALUES[i])
            plot_on_policy(ax, env, B_VALUES[i], NUM_UPDATES[n], COLOURS[i], LINESTYLES[0], NUM_RUNS[n])
            plot_uniform(ax, env, B_VALUES[i], NUM_UPDATES[n], COLOURS[i], LINESTYLES[1], NUM_RUNS[n])
        ax.set(title=str(NUM_STATES[n]) + ' STATES', ylabel='Value of start state under greedy policy', xlabel='Computation time, in expected updates')
        ax.legend()

    plt.show()
