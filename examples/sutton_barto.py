"""
Multi-armed Bandit examples taken from Reinforcement Learning: An Introduction
by Sutton and Barto, 2nd ed. rev Oct2015.
"""
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bandits import (GaussianBandit, Agent, GradientAgent, EpsilonGreedyPolicy,
                     GreedyPolicy, UCBPolicy, SoftmaxPolicy)


class EpsilonGreedyExample:
    label = '2.2 - Action-Value Methods'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, GreedyPolicy()),
        Agent(bandit, EpsilonGreedyPolicy(0.01)),
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
    ]


class OptimisticInitialValueExample:
    label = '2.5 - Optimistic Initial Values'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, GreedyPolicy(), prior=5)
    ]


class UCBExample:
    label = '2.6 - Upper-Confidence-Bound Action Selection'
    bandit = GaussianBandit(10)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(2))
    ]


class GradientExample:
    label = '2.7 - Gradient Bandits'
    bandit = GaussianBandit(10, mu=4)
    policy = SoftmaxPolicy()
    agents = [
        GradientAgent(bandit, policy, alpha=0.1),
        GradientAgent(bandit, policy, alpha=0.4),
        GradientAgent(bandit, policy, alpha=0.1, baseline=False),
        GradientAgent(bandit, policy, alpha=0.4, baseline=False)
    ]


if __name__ == '__main__':
    experiments = 500
    time_steps = 1000

    example = EpsilonGreedyExample
    # example = OptimisticInitialValueExample
    # example = UCBExample
    # example = GradientExample

    bandit = example.bandit
    agents = example.agents

    scores = np.zeros((len(agents), time_steps))
    optimal = np.zeros((len(agents), time_steps))

    for _ in range(experiments):
        bandit.reset()
        for agent in agents:
            agent.reset()
        for t in range(time_steps):
            for i, agent in enumerate(agents):
                action = agent.choose()
                reward, is_optimal = bandit.pull(action)
                agent.observe(reward)

                scores[i, t] += reward
                if is_optimal:
                    optimal[i, t] += 1

    sns.set_style('white')
    plt.subplot(2, 1, 1)
    plt.title(example.label)
    plt.plot(scores.T / experiments)
    plt.ylabel('Average Reward')
    plt.legend(agents, loc=4)
    plt.subplot(2, 1, 2)
    plt.plot(optimal.T / experiments * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Time Step')
    plt.legend(agents, loc=4)
    sns.despine()
    plt.show()
