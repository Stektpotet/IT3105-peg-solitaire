from actorcritic import Actor, Critic
from agent import Agent
from env import Environment


class PegSolitaireAgent(Agent):
    """
    An agent only doing random actions,
    no policy behind them - might be useful for debugging
    """

    def __init__(self, actor: Actor, critic: Critic):
        Agent.__init__(self, actor, critic)

        pass

    def select_action(self, env: Environment):
        possible_actions = self.env.actions()

        pass

    def learn(self, env: Environment, n_episodes: int):
        score = 0.0
        discount = 1.0
        end = False

