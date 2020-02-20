from typing import List

from actorcritic import Actor, Critic
from agent import Agent
from env import Environment
from peg_solitaire.env import PegSolitaireEnvironment
import matplotlib.pyplot as plt

class PegSolitaireAgent(Agent):
    """
    An agent only
    """

    episode_data: [int]

    def __init__(self, actor: Actor, critic: Critic):
        Agent.__init__(self, actor, critic)
        self.episode_data = []

    def track_progression(self, env: PegSolitaireEnvironment, episode: int):
        self.episode_data.append(env.pegs_remaining)

    def plot(self):
        plt.gcf().axes[1].plot(self.episode_data)
