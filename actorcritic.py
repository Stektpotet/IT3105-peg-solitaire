from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Dict

import random
from tensorflow import keras
import tensorflow as tf
from abc import abstractmethod, ABC

class ActorCriticBase(ABC):
    eligibility_traces: Dict

    # TODO: add default values?
    def __init__(self, learning_rate, discount, elig_decay_rate, greed, greed_decay):
        self.learning_rate = learning_rate
        self.discount_rate = discount
        self.eligibility_decay_rate = elig_decay_rate
        self.greed = greed
        self.greed_decay = greed_decay
        self.eligibility_traces = {}

    def reset_eligibility_traces(self):
        self.eligibility_traces.clear()

class ANNModel(ActorCriticBase, ABC):

    @classmethod
    def _make_model(cls, in_shape: tuple, dimensions: tuple, out_shape: tuple):
        inputs = tf.keras.Input(shape=in_shape)
        x = inputs
        for s in dimensions:
            x = tf.keras.layers.Dense(s, activation='relu')(x)
        output = tf.keras.layers.Dense(out_shape, activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=output, name=cls.__name__)

    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, greed, greed_decay):
        ActorCriticBase.__init__(self, learning_rate, discount, elig_decay_rate, greed, greed_decay)

        self.model = self._make_model(state_shape, dimensions, action_shape)
        self.model.summary()
        self.learning_rate = learning_rate
        self.discount_rate = discount


class Actor(ActorCriticBase):

    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, greed, greed_decay):
        ActorCriticBase.__init__(self, learning_rate, discount, elig_decay_rate, greed, greed_decay)

        self.action_shape = action_shape

    def initialize(self, state, actions):
        ActorCriticBase.reset_eligibility_traces(self)
        pass

    @abstractmethod
    def evaluate(self, state, action): pass


class TableActor(Actor):
    state_action_pairs: Dict
    policy: Dict

    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, greed, greed_decay):
        Actor.__init__(self, state_shape, action_shape, dimensions,
                       learning_rate, discount, elig_decay_rate, greed, greed_decay)

        self.state_action_pairs = {}
        self.policy = {}

    def initialize(self, state, actions):
        Actor.initialize(self, state, actions)
        for action in actions:
            self.state_action_pairs[(state, action)] = 0  # TODO: initialize
        pass

    def evaluate(self, state, action):
        if (state, action) not in self.policy:
            self.policy[(state, action)] = 0
            state_action_pairs = 0

        if (state, action) in self.state_action_pairs:
            pass
        pass


class Critic(ActorCriticBase):

    def __init__(self, state_shape, action_shape,
                 learning_rate, discount, elig_decay_rate, greed, greed_decay):
        ActorCriticBase.__init__(self, learning_rate, discount, elig_decay_rate, greed, greed_decay)

        self.action_shape = action_shape
        self.state_shape = state_shape

    def initialize(self, state):
        ActorCriticBase.reset_eligibility_traces(self)
        pass

    @abstractmethod
    def evaluate(self, state): pass


class TableCritic(Critic):

    state_values: Dict

    def __init__(self, state_shape, action_shape,
                 learning_rate, discount, elig_decay_rate, greed, greed_decay):
        Critic.__init__(self, state_shape, action_shape,
                        learning_rate, discount, elig_decay_rate, greed, greed_decay)

        self.state_values = {}

    def initialize(self, state):
        Critic.initialize(self, state)
        self.state_values[state] = random.uniform(0, 0.1)

    def evaluate(self, state):
        pass


class ANNCritic(ANNModel, Critic):
    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, greed, greed_decay):
        Critic.__init__(self, state_shape, action_shape)
        ANNModel.__init__(self, state_shape, action_shape, dimensions,
                          learning_rate, discount, elig_decay_rate, greed, greed_decay)

    def evaluate(self, state):
        pass
