import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plotter(num_runs, steps, agents, bandits, colours, labels):
    average_reward = np.zeros((num_runs, steps, len(agents)))
    best_action_precentage = np.zeros((num_runs, steps, len(agents)))
    for i in tqdm(range(num_runs)):
        for agent in agents:
            agent.reset()
        bandits.reset()

        for j in range(steps):
            best_action = bandits.best_action()
            for a in range(len(agents)):
                action = agents[a].select_action()
                reward = bandits.draw(action)
                agents[a].update_action(action, reward)

                average_reward[i, j, a] = reward
                best_action_precentage[i, j, a] = 100 if action == best_action else 0
            bandits.update()
    
    average_reward = np.average(average_reward, axis=0)
    best_action_precentage = np.average(best_action_precentage, axis=0)

    fig, axs = plt.subplots(2)
    for i in range(len(agents)):
        axs[0].plot(average_reward[:, i], color=colours[i], label=labels[i])
        axs[1].plot(best_action_precentage[:, i], color=colours[i], label=labels[i])
    axs[0].set(ylabel='Average reward', xlabel='Steps')
    axs[1].set(ylabel='Optimal action %', xlabel='Steps')
    axs[0].legend()
    axs[1].legend()
    plt.show()

def parameter_plotter(num_runs, steps, agents, parameters, bandits, colours, labels, include_last_x_steps):
    average_rewards = [np.zeros((num_runs, len(agents[i]))) for i in range(len(agents))]
    for n in tqdm(range(num_runs)):
        for i in range(len(agents)):
            for agent in agents[i]:
                agent.reset()
        bandits.reset()

        rewards = [np.zeros((steps, len(agents[i]))) for i in range(len(agents))]
        for j in tqdm(range(steps), leave=False):
            for i in range(len(agents)):
                for a in range(len(agents[i])):
                    action = agents[i][a].select_action()
                    reward = bandits.draw(action)
                    agents[i][a].update_action(action, reward)

                    rewards[i][j, a] = reward
            bandits.update()
        
        for i in range(len(rewards)):
            rewards[i] = rewards[i][-include_last_x_steps:, :]
            rewards[i] = np.average(rewards[i], axis=0)
            average_rewards[i][n, :] = rewards[i]

    for i in range(len(rewards)):
        average_rewards[i] = np.average(average_rewards[i], axis=0)

    plt.plot()
    for i in range(len(agents)):
        plt.plot(parameters[i], average_rewards[i], color=colours[i], label=labels[i])
    plt.ylabel('Average reward')
    plt.xlabel('Parameter value')
    plt.legend()
    plt.xscale('log', base=2)
    plt.show()
