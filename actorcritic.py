from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow import keras
import tensorflow as tf
from abc import abstractmethod, ABC


class ANNModel(ABC):

    @classmethod
    def _make_model(cls, in_shape: tuple, dimensions: tuple, out_shape: tuple):
        inputs = tf.keras.Input(shape=in_shape)
        x = inputs
        for s in dimensions:
            x = tf.keras.layers.Dense(s, activation='relu')(x)
        output = tf.keras.layers.Dense(out_shape, activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=output, name=cls.__name__)

    def __init__(self, state_shape, action_shape, dimensions, learning_rate, discount_rate):
        self.model = self._make_model(state_shape, dimensions, action_shape)
        self.model.summary()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate


class Actor(ANNModel, ABC):

    def __init__(self, state_shape, action_shape, dimensions, learning_rate, discount_rate):
        ANNModel.__init__(self, state_shape, action_shape, dimensions, learning_rate, discount_rate)


class Critic(ABC):

    def __init__(self, state_shape, action_shape):
        self.action_shape = action_shape
        self.state_shape = state_shape

    @abstractmethod
    def evaluate(self, state, action): pass

class TableCritic(Critic):

    def evaluate(self, state, action):
        pass

    def __init__(self, state_shape, action_shape):
        Critic.__init__(self, state_shape, action_shape)


class ANNCritic(ANNModel, Critic):
    def __init__(self, state_shape, action_shape, dimensions, learning_rate, discount_rate):
        Critic.__init__(self, state_shape, action_shape)
        ANNModel.__init__(self, state_shape, action_shape, dimensions, learning_rate, discount_rate)

    def evaluate(self, state, action):
        pass
