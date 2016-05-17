"""
Takes advantage of multicore systems to speed up the simulation runs.
"""
from bandits.agent import Agent, BetaAgent
from bandits.bandit import BernoulliBandit
from bandits.policy import GreedyPolicy, EpsilonGreedyPolicy, UCBPolicy
from bandits.environment import Environment


if __name__ == '__main__':
    experiments = 500
    trials = 1000

    bandit = BernoulliBandit(10, t=3*1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]
    env = Environment(bandit, agents, label='Bayesian Bandits')
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
    env.plot_beliefs()
