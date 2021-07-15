from tqdm import tqdm

from .misc import *

def semi_gradient_sarsa_lambda(env, agent, gamma, lambda_, max_episodes, alpha, Q, replacing_traces=False, log=True):
    '''
    If replacing_traces is True, then binary features are expected.
    '''
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        state = env.reset()
        action = agent(state, Q)
        z = np.zeros_like(Q.weights)
        terminal = False
        while not terminal:
            next_state, reward, terminal, _ = env.step(action)
            td_error = reward - Q(state, action)
            if replacing_traces:
                z = np.minimum(gamma * lambda_ * z + Q.get_gradients(state, action), 1.0)
            else:
                z = gamma * lambda_ * z + Q.get_gradients(state, action)
            
            if not terminal:
                next_action = agent(next_state, Q)
                td_error += gamma * Q(next_state, next_action)
                state = next_state
                action = next_action
            change = td_error * z
            Q.update_weights(alpha, change)
    return Q

def true_online_sarsa_lambda(env, agent, gamma, lambda_, max_episodes, alpha, Q, log=True):
    '''
    This assumes that Q is linear so that Q.get_gradients(state, action) gives x(state, action).
    '''
    for i in tqdm(range(max_episodes), leave=False, disable=(not log)):
        state = env.reset()
        action = agent(state, Q)
        x = Q.get_gradients(state, action)
        z = np.zeros_like(x)
        q_old = 0
        terminal = False
        while not terminal:
            next_state, reward, terminal, _ = env.step(action)
            next_action = agent(next_state, Q)
            next_x = np.zeros_like(x) if terminal else Q.get_gradients(next_state, next_action)
            q = Q(state, action)
            next_q = 0 if terminal else Q(next_state, next_action)
            td_error = reward + gamma * next_q - q
            z = gamma * lambda_ * z + (1 - alpha * gamma * lambda_ * z.dot(x)) * x
            
            change = (td_error + q - q_old) * z - (q - q_old) * x
            Q.update_weights(alpha, change)

            q_old = next_q
            x = next_x
            action = next_action
            state = next_state
    return Q
