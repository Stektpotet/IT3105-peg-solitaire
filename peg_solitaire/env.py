from typing import Dict

import matplotlib.pyplot as plt

from env import Environment
from peg_solitaire.board import *
from peg_solitaire.board_drawer import BoardDrawer


class PegSolitaireEnvironment(Environment):

    board_drawer: BoardDrawer
    board: Board


    def render(self):
        plt.imshow(self.board_drawer.draw(self.board))
        pass

    def step(self, action: (int, int)) -> (int, bool):
        flat_index, move = action

        pass

    def setup(self, config: Dict):
        env_config = config['env']
        visual_config = config['visual']
        if env_config['type'] == 'triangle':
            self.board = TriangleBoard(env_config['size'])
        else:
            self.board = DiamondBoard(env_config['size'])

        self.board_drawer = BoardDrawer(**visual_config)

        pass

    def set_state(self, state):
        pass

    def reset(self):
        pass

    def score_state(self):
        """
        Naive scoring based on board fullness
        :return: a normalized score
        """
        return self.board.peg_count / self.board.full_count

    def user_modify(self):
        self.render()
        # TODO handle key events
