import numpy as np
import scipy.misc as misc


class MultiArmedBandit(object):
    def __init__(self, k, rewards=None, labels=None):
        self.k = k
        self.rewards = rewards
        self.labels = labels
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    def __init__(self, k, labels=None, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k, labels)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)


class BinomialBandit(MultiArmedBandit):
    """
    Binomial bandits model the probability of a single event occurring given N
    trials i.e. get heads on N coin flips

    Action Value -> [0, 1]
    Reward -> {0, N}
    """
    def __init__(self, k, n, labels=None):
        super(BinomialBandit, self).__init__(k, labels)
        self.n = n
        self.reset()

    def reset(self):
        k = misc.comb(self.n, range(self.n+1))
        self.action_values = np.random.uniform(size=self.k)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):

        return (float(np.random.random() < self.action_values[action]),
                action == self.optimal)


class BernoulliBandit(MultiArmedBandit):
    """
    Bernoulli bandits model the probability of a single event occurring given a
    single trial i.e. get heads on a coin flip

    A bandit arm corresponds to a hit or miss -- either you get a payout or you
    don't.

    Action Value -> [0, 1]
    """
    def __init__(self, k, p=None, rewards=None, labels=None):
        super(BernoulliBandit, self).__init__(k, rewards, labels)
        self.p = p
        self.rewards = np.ones(k)
        self.reset()

    def reset(self):
        if self.p is None:
            self.action_values = np.random.uniform(size=self.k)
        else:
            self.action_values = self.p
        self.optimal = np.argmax(self.action_values*self.rewards)

    def pull(self, action):
        hit = int(np.random.random() < self.action_values[action])
        return hit*self.rewards[action], action == self.optimal


class CategoricalBandit(MultiArmedBandit):
    """
    Categorical bandits model the probability of one of a set of N events
    occurring given a single trial i.e. get j on a k-sided die roll

    Action Value -> {[0, 1], [0, 1], ..., [0, 1]}, sum(AV) = 1
    Reward -> {{0, 1}, {0, 1}, ..., {0, 1}}
    """
    def __init__(self, k, labels=None):
        super(CategoricalBandit, self).__init__(k, labels)
        self.reset()
