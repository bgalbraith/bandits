"""
Multi-armed Bandit examples taken from Reinforcement Learning: An Introduction
by Sutton and Barto, 2nd ed. rev Oct2015.

Takes advantage of multicore systems to speed up the simulation runs.
"""
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np


from bandits.agent import Agent
from bandits.memory import Memory, PreferenceMemory
from bandits.bandit import GaussianBandit
from bandits.strategy import (GreedyStrategy, EpsilonGreedyStrategy,
                              UCBStrategy, SoftmaxStrategy)


class EpsilonGreedyExample:
    label = '2.2 - Action-Value Methods'
    bandit = GaussianBandit(10)
    agents = [
        Agent(Memory(10), GreedyStrategy()),
        Agent(Memory(10), EpsilonGreedyStrategy(0.01)),
        Agent(Memory(10), EpsilonGreedyStrategy(0.1))
    ]


class OptimisticInitialValueExample:
    label = '2.5 - Optimistic Initial Values'
    bandit = GaussianBandit(10)
    agents = [
        Agent(Memory(10), EpsilonGreedyStrategy(0.1)),
        Agent(Memory(10, prior=5), GreedyStrategy())
    ]


class UCBExample:
    label = '2.6 - Upper-Confidence-Bound Action Selection'
    bandit = GaussianBandit(10)
    agents = [
        Agent(Memory(10), EpsilonGreedyStrategy(0.1)),
        Agent(Memory(10), UCBStrategy(2))
    ]


class GradientExample:
    label = '2.7 - Gradient Bandits'
    bandit = GaussianBandit(10, mu=4)
    strategy = SoftmaxStrategy()
    agents = [
        Agent(PreferenceMemory(10, alpha=0.1), strategy),
        Agent(PreferenceMemory(10, alpha=0.4), strategy),
        Agent(PreferenceMemory(10, alpha=0.1, baseline=False), strategy),
        Agent(PreferenceMemory(10, alpha=0.4, baseline=False), strategy)
    ]


def bandit_experiment(bandit, agents, time_steps):
    scores = np.zeros((len(example.agents), time_steps))
    optimal = np.zeros((len(example.agents), time_steps))

    bandit.reset()
    for agent in agents:
        agent.reset()
    for t in range(time_steps):
        for i, agent in enumerate(agents):
            action = agent.choose()
            reward, is_optimal = bandit.pull(action)
            agent.observe(reward)

            scores[i, t] = reward
            if is_optimal:
                optimal[i, t] = 1
    return scores, optimal


if __name__ == '__main__':
    experiments = 2000
    time_steps = 1000

    example = EpsilonGreedyExample
    # example = OptimisticInitialValueExample
    # example = UCBExample
    # example = GradientExample

    bandit = example.bandit
    agents = example.agents

    scores = np.zeros((len(agents), time_steps))
    optimal = np.zeros((len(agents), time_steps))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        res = [pool.apply_async(bandit_experiment, (bandit, agents, time_steps))
               for i in range(experiments)]
        for r in res:
            s, o = r.get(1)
            scores += s
            optimal += o

    plt.subplot(2, 1, 1)
    plt.title(example.label)
    plt.plot(scores.T / experiments)
    plt.ylim(0, 1.6)
    plt.ylabel('Average Reward')
    plt.legend(agents, loc=4)
    plt.subplot(2, 1, 2)
    plt.plot(optimal.T / experiments * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Time Step')
    plt.legend(agents, loc=4)
    plt.show()
