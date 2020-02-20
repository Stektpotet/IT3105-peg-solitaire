from abc import ABC, abstractmethod
from typing import Dict, List


class Environment(ABC):

    @abstractmethod
    def setup(self, config: Dict): pass

    @abstractmethod
    def step(self, action) -> (int, bool):
        """
        Perform an action in the environment
        :param action: the action to perform
        :return: (int reward, bool done)
        """
        pass

    @abstractmethod
    def render(self): pass

    @abstractmethod
    def set_state(self, state): pass

    @abstractmethod
    def reset(self): pass

    @property
    @abstractmethod
    def state_key(self): pass

    @abstractmethod
    def has_won(self): pass

    @abstractmethod
    def actions(self) -> List:
        """
        :return: A list of the feasible actions given the current state
        """
        pass
