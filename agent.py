import numpy as np
from tensorflow import keras
import tensorflow as tf
from abc import ABC, abstractmethod
from env import Environment
# from agent import Environment

class RandomAgent(ABC):
    def __init__(self, env: Environment, actor, critic):
        self.env = env
        pass

    # @abstractmethod
    def select_action(self, state):
        pass

    def play(self, env):

        score = 0.0
        discount = 1.0
        end = False


class Agent(ABC):
    def __init__(self, env: Environment, actor, critic):
        self.env = env
        pass

    # @abstractmethod
    def select_actions(self, state): pass

    def play(self, env):

        score = 0.0
        discount = 1.0
        end = False
