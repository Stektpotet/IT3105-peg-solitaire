from abc import ABC, abstractmethod
from typing import Dict, List


class Environment(ABC):

    @abstractmethod
    def setup(self, config: Dict):
        """
        Perform setup of the environment based on the content of a config dictionary
        :param config: the configuration
        :return:
        """
        pass

    @abstractmethod
    def step(self, action) -> (float, bool):
        """
        Perform an action in the environment
        :param action: the action to perform
        :return: (int reward, bool done)
        """
        pass

    @abstractmethod
    def render(self):
        """
        Perform a visualization of the environment.
        :return:
        """
        pass

    @abstractmethod
    def set_state(self, state: bytes):
        """
        Put the environment in the specified state
        :param state: the state to enter
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        :return: Put the environment back into it's original configuration
        """
        pass

    @property
    @abstractmethod
    def state_key(self) -> bytes:
        """
        :return: A binary representation (as it's hashable, i.e. useful for Dict)
        of the current state.
        """
        pass

    @abstractmethod
    def has_won(self) -> bool:
        """
        :return: True if the goal of the environment is reached
        """
        pass

    @abstractmethod
    def actions(self) -> List:
        """
        :return: A list of the feasible actions given the current state
        """
        pass
