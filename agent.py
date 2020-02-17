import random

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
        pass

    @abstractmethod
    def select_actions(self):
        """
        Select an action based on the actors policy
        :return:
        """
        # for action in self.env.actions():
        #     self.actor.evaluate(self.env.state_key, action)
        pass

    def learn(self, env: Environment, n_episodes: int):
        episode_score = 0.0
        end = False

        # START OF ACTOR-CRITIC ALGORITHM #

        self.critic.initialize(env.state_key)
        self.actor.initialize(env.state_key, env.actions())

        def episode_rollout(episode: int):
            env.reset()

            self.actor.reset_eligibility_traces()
            self.critic.reset_eligibility_traces()

            while True:
                action = self.select_actions()
                reward, done = env.step(action)

                if done:
                    print("Reached end-state!")
                    if reward > 0:
                        print("VICTORY!")
                    # TODO: check victory
                    return reward

        for i in range(n_episodes):
            episode_rollout(i)

        print(f"Learning stopped! {n_episodes} episodes completed")


        # TODO: Implement the agent actions
        # https://github.com/karl-hajjar/RL-solitaire/blob/8386fe857f902c83c21a9addc5d6e6336fc9c66a/agent.py#L113
        # for inspiration

    def train(self, env, n_games):
        # TODO: We can scale up the training to utilize multiple threads multiple environments at once
        # To do so, we'll need to decouple env from agent, i.e. don't hold a reference
        pass

class RandomAgent(Agent):
    """
    An agent only doing random actions,
    no policy behind them - might be useful for debugging
    """

    def select_actions(self):
        possible_actions = self.env.actions()
        return random.choice(possible_actions)
        pass

    def __init__(self, env: Environment, actor: Actor, critic: Critic):
        Agent.__init__(self, env, actor, critic)

        pass


class PegSolitaireAgent(Agent):
    """
    An agent only doing random actions,
    no policy behind them - might be useful for debugging
    """

    def __init__(self, env: Environment, actor: Actor, critic: Critic):
        Agent.__init__(self, env, actor, critic)

        pass

    # @abstractmethod
    def select_action(self):

        self.env.actions()

        pass

    def learn(self, env):
        score = 0.0
        discount = 1.0
        end = False

