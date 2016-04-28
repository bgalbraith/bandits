import numpy as np


class GreedyStrategy(object):
    def __str__(self):
        return 'greedy'

    def choose(self, memory, time_step):
        action = np.argmax(memory.value_estimates)
        check = np.where(memory.value_estimates == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)


class EpsilonGreedyStrategy(GreedyStrategy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, memory, time_step):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(memory.value_estimates))
        else:
            return super(EpsilonGreedyStrategy, self).choose(memory, time_step)


class UCBStrategy(object):
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, memory, time_step):
        exploration = np.log(time_step+1) / memory.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = memory.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)


class SoftmaxStrategy(object):
    def __str__(self):
        return 'SM'

    def choose(self, memory, time_step):
        a = memory.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a))
        cdf = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]
