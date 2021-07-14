from numpy.core.overrides import verify_matching_signatures
from tqdm import tqdm

from .misc import *

def offline_lambda_return_v(env, agent, gamma, lambda_, max_episodes, alpha, V, log=True):
    '''
    This implementation gives slightly different results than the one in the book. When doing offline lambda-
    return, you have to choose whether to calculate all of the targets first or after each semi-gradient update.
    The latter requires more computation and both have drawbacks (this one uses the former). The drawback of the
    former is that you won't use your latest weights when computing the targets, meaning that lambda=0 does not result
    in TD(0). The latter means that rewards can be used to update a prediction later used for bootstrapping. This
    is problematic because if you bootstrap from a state that has already been updated (more general when not using
    tabular), you also use the rewards in front of it as part of your update. This means that rewards that happened
    during the trajectory (variance) can make their way into bootstrapping updates, which harms us more for larger
    lambdas. It is like you are using information (in form of rewards) that should not be available to you. 
    Compare this to truncated lambda-returns which do not have this problem.
    Finally, the true online-TD method gets the best of both worlds. When bootstrapping, it uses the weights updated
    as much as possible without using information past that state. This means lambda=0 -> TD(0) and you don't get
    the drawback of the latter method. 
    '''
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        states = [env.reset()]
        actions = []
        rewards = []
        terminal = False
        while not terminal:
            actions.append(agent(states[-1]))
            next_state, reward, terminal, _ = env.step(actions[-1])
            if not terminal:
                states.append(next_state)
            rewards.append(reward)
        assert len(states) == len(actions) == len(rewards), 'The sequence of experience have different lengths'
        
        Gs = [0] * (len(states) + 1)
        for i in reversed(range(len(states))):
            Gs[i] = rewards[i] + lambda_ * gamma * Gs[i + 1]
            if i + 1 < len(states):
                Gs[i] += gamma * (1 - lambda_) * V(states[i + 1])
        for i in range(len(states)):
            change = (Gs[i] - V(states[i])) * V.get_gradients(states[i])
            V.update_weights(alpha, change)
    return V

def TD_lambda_v(env, agent, gamma, lambda_, max_episodes, alpha, V, log=True):
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        state = env.reset()
        z = np.zeros_like(V.weights)
        terminal = False
        while not terminal:
            action = agent(state)
            next_state, reward, terminal, _ = env.step(action)
            z = gamma * lambda_ * z + V.get_gradients(state)
            td_error = reward - V(state)
            if not terminal:
                td_error += gamma * V(next_state)
            V.update_weights(alpha, td_error * z)
            state = next_state
    return V

def online_lambda_return_v(env, agent, gamma, lambda_, max_episodes, alpha, V, log=True):
    '''
    This implementation makes heavy use of eq. 12.10 in the book in order to incrementally
    increase the horizon. Therefore, the computation per timestep is reduced to being
    linear in the timestep (although the updates are for function approximation O(t*d)).
    '''
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        start_weights = V.weights
        last_weights = V.weights
        Gs = []

        states = [env.reset()]
        actions = []
        rewards = []
        terminal = False
        while not terminal:
            actions.append(agent(states[-1]))
            next_state, reward, terminal, _ = env.step(actions[-1])
            states.append(next_state) # This also adds the terminal state for once. A check is done below.
            rewards.append(reward)

            v_state = V.get_target(last_weights, states[-2])
            v_next_state = 0 if terminal else V(states[-1])
            td_error = rewards[-1] + gamma * v_next_state - v_state
            Gs.append(v_state)
            gamma_lambda_multiplier = 1
            for i in reversed(range(len(Gs))):
                Gs[i] += gamma_lambda_multiplier * td_error
                gamma_lambda_multiplier *= gamma * lambda_
            
            last_weights = V.weights
            V.weights = start_weights
            for i in range(len(Gs)):
                change = (Gs[i] - V(states[i])) * V.get_gradients(states[i])
                V.update_weights(alpha, change)
    return V

def true_online_TD_lambda_v(env, agent, gamma, lambda_, max_episodes, alpha, V, log=True):
    '''
    This is currently not used in any experiments, but is here for reference. Feel free to run
    it together with online_lambda_return_v to show empirically that they are equivalent.
    This assumes that V is linear so that V.get_gradients(state) gives x(state).
    '''
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        state = env.reset()
        x = V.get_gradients(state)
        z = np.zeros_like(x)
        v_old = 0
        terminal = False
        while not terminal:
            action = agent(state)
            next_state, reward, terminal, _ = env.step(action)
            next_x = np.zeros_like(x) if terminal else V.get_gradients(next_state)

            v = V(state)
            next_v = 0 if terminal else V(next_state)
            delta = reward + gamma * next_v - v
            z = gamma * lambda_ * z + (1 - alpha * gamma * lambda_ * z.dot(x)) * x
            change = (delta + v - v_old) * z - (v - v_old) * x
            V.update_weights(alpha, change)
            v_old = next_v
            x = next_x
            state = next_state
    return V
