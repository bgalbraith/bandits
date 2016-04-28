class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self, memory, strategy):
        self.memory = memory
        self.strategy = strategy
        self.t = 0

    def __str__(self):
        return '{}/{}'.format(str(self.memory), str(self.strategy))

    def choose(self):
        action = self.strategy.choose(self.memory, self.t)
        self.memory.last_action = action
        return action

    def observe(self, reward):
        self.memory.update(reward)
        self.t += 1

    def reset(self):
        self.memory.reset()
        self.t = 0
