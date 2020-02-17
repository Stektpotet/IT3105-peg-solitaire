import random

import numpy as np
from tensorflow import keras
import tensorflow as tf
from abc import ABC, abstractmethod

from actorcritic import Actor, Critic
from env import Environment


# from agent import Environment

class Agent(ABC):
    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic
        pass

    def select_actions(self, env: Environment):
        """
        Select an action based on the actors policy
        :return:
        """
        for action in env.actions():
            self.actor.evaluate(env.state_key, action)
        pass

    def _episode_rollout(self, env: Environment, episode: int):
        env.reset()
        self.actor.reset_eligibility_traces()
        self.critic.reset_eligibility_traces()

        while True:
            action = self.select_actions(env)
            reward, done = env.step(action)

            if done:
                print(f"Reached end-state: {reward}")
                if reward == 1:
                    print("VICTORY!")
                # TODO: check victory
                return reward

    def learn(self, env: Environment, n_episodes: int):

        episode_score = 0.0
        end = False

        # START OF ACTOR-CRITIC ALGORITHM #

        self.critic.initialize(env.state_key)
        self.actor.initialize(env.state_key, env.actions())

        for i in range(n_episodes):
            self._episode_rollout(env, i)

        print(f"Learning stopped! {n_episodes} episodes completed")

        # TODO: Implement the agent actions
        # https://github.com/karl-hajjar/RL-solitaire/blob/8386fe857f902c83c21a9addc5d6e6336fc9c66a/agent.py#L113
        # for inspiration

    def train(self, env: Environment, n_games: int):
        # TODO: We can scale up the training to utilize multiple threads multiple environments at once
        # To do so, we'll need to decouple env from agent, i.e. don't hold a reference
        pass


class RandomAgent(Agent):
    """
    An agent only doing random actions,
    no policy behind them - might be useful for debugging
    """

    def select_actions(self, env: Environment):
        possible_actions = env.actions()
        return random.choice(possible_actions)
        pass

    def __init__(self, actor: Actor, critic: Critic):
        Agent.__init__(self, actor, critic)

        pass
