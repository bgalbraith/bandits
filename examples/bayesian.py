"""
Takes advantage of multicore systems to speed up the simulation runs.
"""
import matplotlib
matplotlib.use('qt4agg')
from bandits.agent import Agent, BetaAgent
from bandits.bandit import BernoulliBandit, BinomialBandit
from bandits.policy import GreedyPolicy, EpsilonGreedyPolicy, UCBPolicy
from bandits.environment import Environment


class BernoulliExample:
    label = 'Bayesian Bandits - Bernoulli'
    bandit = BernoulliBandit(10, t=3*1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]


class BinomialExample:
    label = 'Bayesian Bandits - Binomial (n=5)'
    bandit = BinomialBandit(10, n=5, t=3*1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]


if __name__ == '__main__':
    experiments = 500
    trials = 1000

    example = BernoulliExample()
    # example = BinomialExample()

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
    env.plot_beliefs()
