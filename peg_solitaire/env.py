from typing import Dict
from copy import deepcopy
import matplotlib.pyplot as plt

from env import Environment
from peg_solitaire.board import *
from peg_solitaire.board_drawer import BoardDrawer


class PegSolitaireEnvironment(Environment):

    board_drawer: BoardDrawer
    board: Board
    initial_board: Board  # A copy of the board in its initial state (pre-training)

    should_render: bool

    def render(self):
        plt.imshow(self.board_drawer.draw(self.board))
        # MORE?
        #  plt pause or something?
        pass

    def step(self, action: (int, int)) -> (int, bool):
        flat_index, move = action
        # TODO: apply the move to the board
        # For decopled code, it might be worth just passing it on to the board :thinkin:
        pass

    # NOTE: we want to allow creation before setup, hence this is not in __init__
    # Though if need be we may put it there later... :thinking:
    def setup(self, config: Dict):
        env_config = config['env']
        visual_config = config['visual']
        if env_config['type'] == 'triangle':
            self.board = TriangleBoard(env_config['size'])
        else:
            self.board = DiamondBoard(env_config['size'])

        self.board_drawer = BoardDrawer(**visual_config)

        # More?
        pass

    def set_state(self, state: np.ndarray):
        self.board.pegs = state
        # More?
        pass

    def reset(self):
        self.board = deepcopy(self.initial_board)
        # More?
        pass

    def score_state(self):
        return self._score_state()
    @staticmethod
    def _score_state(board: Board):
        """
        Naive scoring based on board fullness
        :return: a normalized score
        """
        return board.peg_count / board.full_count

    def user_modify(self):
        self.render()
        # TODO input loop allowing user to modify board - needs to render for every event
        self.initial_board = deepcopy(self.board)  # Store aside the board in its starting config

    def generate_state_action_pairs(self):
        state_action_pairs = {}

        def step(board):
            for p in zip(*np.where(board.pegs == 0)):  # For each open position - TODO: verify that masked values are not used
                p_flat = board.index_flat(p)
                for move in board.valid_moves(p):
                    print(move)
                    if board.valid_action(p_flat, move):
                        b = deepcopy(board)
                        b.apply_action(p_flat, move)
                        state_action_pairs[(p_flat, move)] = self._score_state(b)  # Add state_action_pair
                        step(b)

        step(deepcopy(self.initial_board))
        return state_action_pairs
