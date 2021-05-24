class Agent(object):
    def __init__(self, k, action_selector, q_approximator):
        self.action_selector = action_selector
        self.q_approximator = q_approximator

    def select_action(self):
        return self.action_selector(self.q_approximator.get())

    def update_action(self, action, value):
        self.q_approximator.update_action(action, value)

    def reset(self):
        self.action_selector.reset()
        self.q_approximator.reset()
