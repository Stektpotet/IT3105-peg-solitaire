import numpy as np
from tensorflow import keras
import tensorflow as tf
from abc import ABC, abstractmethod

from actorcritic import Actor, Critic
from env import Environment
# from agent import Environment

class Agent(ABC):
    def __init__(self, env: Environment, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.epsilon_greed = 1
        pass



    # @abstractmethod
    def select_actions(self, state):
        rand = np.random.rand()

        if rand < self.epsilon_greed:
            # Explore, i.e. do random action
            pass
        else:
            # select best action
            pass

        pass

    def play(self, env):

        episode_score = 0.0
        discount = 1.0
        end = False

        # TODO: Implement the agent actions
        # https://github.com/karl-hajjar/RL-solitaire/blob/8386fe857f902c83c21a9addc5d6e6336fc9c66a/agent.py#L113
        # for inspiration

class RandomAgent(Agent):
    """
    An agent only doing random actions,
    no policy behind them - might be useful for debugging
    """
    def __init__(self, env: Environment, actor: Actor, critic: Critic):
        Agent.__init__(self, env, actor, critic)

        pass

    # @abstractmethod
    def select_action(self, state):
        # This won't really work, the agent needs to know the action space too (i.e. min max)
        return np.random.rand(self.actor.action_shape)
        pass

    def play(self, env):

        score = 0.0
        discount = 1.0
        end = False


class PegSolitaireAgent(Agent):
    """
    An agent only doing random actions,
    no policy behind them - might be useful for debugging
    """
    def __init__(self, env: Environment, actor: Actor, critic: Critic):
        Agent.__init__(self, env, actor, critic)

        pass

    # @abstractmethod
    def select_action(self, state):
        pass

    def play(self, env):

        score = 0.0
        discount = 1.0
        end = False

