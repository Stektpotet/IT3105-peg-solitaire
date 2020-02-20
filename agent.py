import random
import time

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
from abc import ABC, abstractmethod

from actorcritic import Actor, Critic, TableCritic
from env import Environment


# from agent import Environment

class Agent(ABC):
    def __init__(self, actor: Actor, critic: Critic):
        self.actor = actor
        self.critic = critic
        pass

    @abstractmethod
    def track_progression(self, env: Environment, episode: int): pass

    @abstractmethod
    def plot(self): pass

    def select_actions(self, env: Environment):
        """
        Select an action based on the actors policy
        :return:
        """
        # TODO: Restructure this - combine actor and critic as class
        return self.actor.select_action(env.state_key, env.actions())
        pass

    # NOTE: Large portions of this function could be simplified by combining actor and critic in one class
    def _episode_rollout(self, env: Environment, episode: int):
        env.reset()
        self.actor.reset_eligibility_traces()
        self.critic.reset_eligibility_traces()

        state = env.state_key
        action = self.select_actions(env)
        while True:
            # 1 DO ACTION a FROM STATE s, MOVING THE SYSTEM TO STATE s’ AND RECEIVING REINFORCEMENT
            reward, done = env.step(action)

            if done:
                print(f"Reached end-state: {reward}")
                return env.has_won()

            # 2. ACTOR: a’ ← Π(s’) THE ACTION DICTATED BY THE CURRENT POLICY FOR STATE s’.
            state_prime = env.state_key
            action_prime = self.select_actions(env)

            # 3. ACTOR: e(s,a) ← 1 (the actor keeps SAP-based eligibilities)
            self.actor.set_eligibility_of_state_action_pair(state, action)

            # Step 4 through 6 can be moved into one call on critic :tinking:
            # 4. CRITIC: δ ← r +γV(s')−V(s)
            error = self.critic.error(state, state_prime, reward)



            # 5. CRITIC: e(s) ← 1 (the critic needs state-based eligibilities)
            self.critic.set_eligibility_of_state(state)

            # 6
            self.critic.update_all(error, state, state_prime, reward)
            self.actor.update_all(error)

            # 7
            state = state_prime
            action = action_prime


    def learn(self, env: Environment, n_episodes: int):
        wins = 0

        # START OF ACTOR-CRITIC ALGORITHM #

        self.critic.initialize(env.state_key)
        self.actor.initialize(env.state_key, env.actions())

        for i in tqdm(range(n_episodes)):
            if self._episode_rollout(env, i):
                wins += 1
            self.track_progression(env, i)

        self.plot()
        print(f"Learning stopped! {n_episodes} episodes completed\n\twins: {wins}")
        print("Curiosity at end: ", self.actor.curiosity)

        # TODO: Implement the agent actions
        # https://github.com/karl-hajjar/RL-solitaire/blob/8386fe857f902c83c21a9addc5d6e6336fc9c66a/agent.py#L113
        # for inspiration

    def _test_episode_rollout(self, env: Environment, episode: int):
        env.reset()
        self.actor.reset_eligibility_traces()
        self.critic.reset_eligibility_traces()
        reward = 0

        done = False
        while not done:
            reward, done = env.step(self.select_actions(env))
        return env.has_won()

    def test(self, env: Environment, n_games):
        env.render()
        wins = 0
        self.actor.curiosity = 0
        for i in tqdm(range(n_games)):
            if self._test_episode_rollout(env, i):
                wins += 1
        env.render()
        print(f"Testing stopped... \n\twins: {100*wins/n_games}% success when greedy")

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
