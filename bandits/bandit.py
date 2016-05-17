import numpy as np
import pymc3 as pm


class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu=0, sigma=1):
        super(GaussianBandit, self).__init__(k)
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
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.

    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """
    def __init__(self, k, n, p=None, t=None):
        super(BinomialBandit, self).__init__(k)
        self.n = n
        self.p = p
        self.t = t
        self.model = pm.Model()
        with self.model:
            self.bin = pm.Binomial('binomial', n=n*np.ones(k, dtype=np.int),
                                   p=np.ones(k)/n, shape=(1, k), transform=None)
        self._samples = None
        self._cursor = 0

        self.reset()

    def reset(self):
        if self.p is None:
            self.action_values = np.random.uniform(size=self.k)
        else:
            self.action_values = self.p
        self.bin.distribution.p = self.action_values
        if self.t is not None:
            self._samples = self.bin.random(size=self.t).squeeze()
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        return self.sample[action], action == self.optimal

    @property
    def sample(self):
        if self._samples is None:
            return self.bin.random()
        else:
            val = self._samples[self._cursor]
            self._cursor += 1
            return val


class BernoulliBandit(BinomialBandit):
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This is
    the special case of the Binomial distribution where N=1.

    In the bandit scenario, this can be used to approximate a hit or miss event,
    such as if a user clicks on a headline, ad, or recommended product.
    """
    def __init__(self, k, p=None, t=None):
        super(BernoulliBandit, self).__init__(k, 1, p=p, t=t)
