from abc import ABC, abstractmethod
from typing import Dict


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

    # @abstractmethod
    # def set_render_target(self, tk.Canvas):