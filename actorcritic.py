import time
from timeit import Timer
from typing import Dict

import numpy as np
import random
from tensorflow import uint8
import tensorflow as tf
from abc import abstractmethod, ABC

class ActorCriticBase(ABC):

    # TODO: add default values?
    def __init__(self, learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        self.learning_rate = learning_rate
        self.discount = discount
        self.eligibility_decay_rate = elig_decay_rate
        self.curiosity = curiosity
        self.curiosity_decay = curiosity_decay

    @abstractmethod
    def reset_eligibility_traces(self): pass

    @abstractmethod
    def update_all(self, error, *args): pass

    def __repr__(self):
        return f"<{type(self)}\n" \
               f"lr: {self.learning_rate}\n" \
               f"gamma: {self.discount}\n" \
               f"epsilon: {self.curiosity}>"

class ANNModel(ActorCriticBase, ABC):

    @classmethod
    def _make_model(cls, in_shape: tuple, dimensions: tuple, out_shape: tuple):
        inputs = tf.keras.Input(shape=in_shape)
        x = inputs
        for s in dimensions:
            x = tf.keras.layers.Dense(s, activation="sigmoid")(x)
        output = tf.keras.layers.Dense(out_shape, activation="linear")(x)
        return tf.keras.Model(inputs=inputs, outputs=output, name=cls.__name__)

    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        ActorCriticBase.__init__(self, learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)

        self.model = self._make_model(state_shape, dimensions, action_shape)
        self.model.summary()
        self.learning_rate = learning_rate
        self.discount_rate = discount


class Actor(ActorCriticBase):

    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        ActorCriticBase.__init__(self, learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)
        self.eligibility_traces = {}

        self.action_shape = action_shape

    def initialize(self, state, actions):
        ActorCriticBase.reset_eligibility_traces(self)
        pass

    @abstractmethod
    def select_action(self, state, actions): pass


class TableActor(Actor):
    policy: Dict

    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        Actor.__init__(self, state_shape, action_shape, dimensions,
                       learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)

        self.policy = {}

    def initialize(self, state, actions):
        Actor.initialize(self, state, actions)
        for action in actions:
            self.policy[(state, action)] = 0  # TODO: initialize
        pass

    def update_all(self, error, *args):
        for sap in self.eligibility_traces.keys():
            self.policy[sap] += self.learning_rate * error * self.eligibility_traces[sap]
            self.eligibility_traces[sap] *= self.discount * self.eligibility_decay_rate

    def select_action(self, state, actions):
        local_policy_mapping = {}

        for action in actions:
            sap = (state, action)

            if sap not in self.policy:
                self.policy[sap] = 0
            local_policy_mapping[sap] = self.policy[sap]
            if sap not in self.eligibility_traces:
                self.eligibility_traces[sap] = 0

        # An ε-greedy strategy makes a random choice of actions with probability ε,
        # and the greedy choice with probability 1−ε.

        # Normalize for some reason
        # local_policy_mapping = {
        #     key: value / sum(local_policy_mapping.values()) for key, value in local_policy_mapping.items()
        # }

        # If more curious than greedy; explore!
        if random.uniform(0, 1) > self.curiosity:
            selected_action = max(local_policy_mapping, key=local_policy_mapping.get)[1]
        else:
            selected_action = random.choice(actions)
        self.curiosity *= self.curiosity_decay
        return selected_action

    def reset_eligibility_traces(self):
        self.eligibility_traces.clear()
        pass


class Critic(ActorCriticBase):

    def __init__(self, state_shape, action_shape,
                 learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        ActorCriticBase.__init__(self, learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)
        self.action_shape = action_shape
        self.state_shape = state_shape

    def initialize(self, state):
        ActorCriticBase.reset_eligibility_traces(self)
        pass

    @abstractmethod
    def error(self, state, state_prime, reward): pass


class TableCritic(Critic):

    state_values: Dict

    def __init__(self, state_shape, action_shape,
                 learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        Critic.__init__(self, state_shape, action_shape,
                        learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)
        self.eligibility_traces = {}
        self.state_values = {}

    def initialize(self, state):
        Critic.initialize(self, state)
        self.state_values[state] = random.uniform(0, 0.1)

    def error(self, state, state_prime, reward):
        if state not in self.state_values:  # Note: initial values may play a big part :thinking:
            self.state_values[state] = random.uniform(0.49, 0.5)  # This random distribution seems to work well
        if state_prime not in self.state_values:
            self.state_values[state_prime] = random.uniform(0.49, 0.)

        if state not in self.eligibility_traces:  # NOTE: this can be set to 1 immediately according to the algorithm
            self.eligibility_traces[state] = 0
        if state_prime not in self.eligibility_traces:
            self.eligibility_traces[state_prime] = 0

        return reward + self.discount * self.state_values[state_prime] - self.state_values[state]

    def reset_eligibility_traces(self):
        self.eligibility_traces.clear()

    def update_all(self, error, *args):
        for s in self.eligibility_traces.keys():
            self.state_values[s] += self.learning_rate * error * self.eligibility_decay_rate
            self.eligibility_traces[s] *= self.discount * self.eligibility_decay_rate


class ANNCritic(ANNModel, Critic):
    def __init__(self, state_shape, action_shape, dimensions,
                 learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay):
        Critic.__init__(self, state_shape, action_shape,
                        learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)
        ANNModel.__init__(self, state_shape, 1, dimensions,
                          learning_rate, discount, elig_decay_rate, curiosity, curiosity_decay)

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adagrad(learning_rate)
        )
        self.eligibility_traces = np.zeros(len(self.model.trainable_variables))

    def reset_eligibility_traces(self):
        self.eligibility_traces = np.zeros_like(self.model.trainable_variables)

    def error(self, state, state_prime, reward):
        s_ = tf.convert_to_tensor(np.expand_dims(np.frombuffer(state_prime, dtype=np.uint8), axis=0), dtype=uint8)
        s = tf.convert_to_tensor(np.expand_dims(np.frombuffer(state, dtype=np.uint8), axis=0), dtype=uint8)
        return (reward + self.discount * self.model(s_)[0, 0] - self.model(s)[0, 0]).numpy()

    def update_all(self, error, *args):
        """
        Trains the model
        Updates trainable variables (weights and biases) + eligibility traces
        :param error:
        :return:
        """
        state, next_state, reward = args

        with tf.GradientTape() as tape:
            s_ = tf.convert_to_tensor(np.expand_dims(np.frombuffer(next_state, dtype=np.uint8), axis=0), dtype=uint8)
            s = tf.convert_to_tensor(np.expand_dims(np.frombuffer(state, dtype=np.uint8), axis=0), dtype=uint8)
            target = reward + self.discount * self.model(s_)

            prediction = self.model(s)
            loss = self.model.loss(target, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        #print(f"trainable_variables {self.model.trainable_variables}")

        # For all the gradient layers' matrices:
        # each element g is here -2δ(∂V(s)/∂w)
        for i, g in enumerate(gradients):
            # print(f" elg_trace: {self.eligibility_traces[i]}")
            # print(f" discount: {self.discount}")
            # print(f" elg_decay: {self.eligibility_decay_rate}")
            # print(f"g is : {g}")

            # decay self
            self.eligibility_traces[i] *= self.discount * self.eligibility_decay_rate
            # add gradient - read eq. 16 & 21 (SDG  elig update L3-Slide8)
            self.eligibility_traces[i] += g / (2 * error)
            # update gradient - it's later used in the weight update performed by the optimizer
            # w_ = w + α δ e (w_ = w + learning rate * error * eligibility)
            # modify gradient (∂L/∂w) to δ*elig (read eq. 16 & 21)
            gradients[i] = self.eligibility_traces[i] * error

        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


        #print(self.model.optimizer.get_gradients(loss, self.model.trainable_variables))
