"""
Memory tracks two things:
  1) how many times an arm is chosen
  2) expected reward value per arm
"""
import numpy as np
import scipy.stats as stats


class Memory(object):
    def __init__(self, k, prior=0):
        self.k = k
        self.prior = prior
        self.value_estimates = prior*np.ones(k)
        self.action_attempts = np.zeros(k)
        self.last_action = None

    def __str__(self):
        return 'f'

    def reset(self):
        self.value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None

    def update(self, reward):
        self.action_attempts[self.last_action] += 1

        n = self.action_attempts[self.last_action]
        q = self.value_estimates[self.last_action]

        self.value_estimates[self.last_action] += (1/n)*(reward - q)


class PreferenceMemory(Memory):
    def __init__(self, k, prior=0, alpha=0.1, baseline=True):
        super(PreferenceMemory, self).__init__(k, prior)
        self.alpha = alpha
        self.baseline = baseline
        self.average_reward = 0

    def __str__(self):
        return 'fp'

    def reset(self):
        super(PreferenceMemory, self).reset()
        self.average_reward = 0

    def update(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.baseline:
            diff = reward - self.average_reward
            self.average_reward += 1/np.sum(self.action_attempts) * diff

        pi = np.exp(self.value_estimates) / np.sum(np.exp(self.value_estimates))

        ht = self.value_estimates[self.last_action]
        ht += self.alpha*(reward - self.average_reward)*(1-pi[self.last_action])
        self.value_estimates -= self.alpha*(reward - self.average_reward)*pi
        self.value_estimates[self.last_action] = ht


class BetaMemory(object):
    def __init__(self, k):
        self.k = k
        self.alpha = np.ones(k)
        self.beta = np.ones(k)
        self._reward_estimates = np.ones(self.k)
        self._value_estimates = np.zeros(k)
        self.action_attempts = np.zeros(k)
        self.last_action = None

    def __str__(self):
        return 'b'

    def reset(self):
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)
        self._reward_estimates = np.ones(self.k)
        self._value_estimates[:] = 0
        self.action_attempts[:] = 0
        self.last_action = None

    def update(self, reward):
        self.action_attempts[self.last_action] += 1
        if reward > 0:
            self._reward_estimates[self.last_action] = reward
            x = 1
        else:
            x = 0

        self.alpha[self.last_action] += x
        self.beta[self.last_action] += 1 - x
        self._value_estimates = stats.beta.ppf(np.random.random(), self.alpha,
                                               self.beta)
        self._value_estimates *= self._reward_estimates

    @property
    def value_estimates(self):
        return self._value_estimates


class DirichletMemory(object):
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.alpha = np.ones((n, k))
        self._reward_estimates = np.ones(self.k)
        self._value_estimates = np.zeros(k)
        self.action_attempts = np.zeros(k)
        self.last_action = None

    def __str__(self):
        return 'b'

    def reset(self):
        self.alpha = np.ones((self.n, self.k))
        self._reward_estimates = np.ones(self.k)
        self._value_estimates[:] = 0
        self.action_attempts[:] = 0
        self.last_action = None

    def update(self, reward):
        self.action_attempts[self.last_action] += 1
        if reward > 0:
            self._reward_estimates[self.last_action] = reward
            x = 1
        else:
            x = 0

        self.alpha[self.last_action] += x
        self._value_estimates = stats.dirichlet.ppf(np.random.random(),
                                                    self.alpha)
        self._value_estimates *= self._reward_estimates

    @property
    def value_estimates(self):
        return self._value_estimates
