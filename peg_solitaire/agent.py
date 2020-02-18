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
