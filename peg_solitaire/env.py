from typing import Dict
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

from env import Environment
from peg_solitaire.board import *
from peg_solitaire.board_drawer import BoardDrawer


class PegSolitaireEnvironment(Environment):

    board_drawer: BoardDrawer
    board: Board
    initial_board: Board  # A copy of the board in its initial state (pre-training)

    should_render: bool

    def _render(self, board):
        plt.clf()
        plt.imshow(self.board_drawer.draw(board))
        plt.pause(0.01)

    def render(self):
        self._render(self.board)
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

        matplotlib.use(backend="TkAgg")
        self.fig, self.ax = plt.subplots()
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
        return (board.full_count - board.peg_count - 1) / board.full_count

    def user_modify(self):
        self.render()
        wait_for_enter_key = False

        def on_click(event):
            print(self.board_drawer.index_from_screen_space(self.board, event.x, event.y))
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))

        cid_mousepress = self.fig.canvas.mpl_connect('button_press_event', on_click)

        def on_key_press(event: KeyEvent):
            nonlocal wait_for_enter_key
            print(event.key)
            if event.key == "enter":
                wait_for_enter_key = False
        cid_keypress = self.fig.canvas.mpl_connect('key_press_event', on_key_press)

        while wait_for_enter_key:
            plt.waitforbuttonpress(timeout=100)
            pass
            # TODO input loop allowing user to modify board - needs to render for every event

        self.fig.canvas.mpl_disconnect(cid_mousepress)
        self.fig.canvas.mpl_disconnect(cid_keypress)

        self.initial_board = deepcopy(self.board)  # Store aside the board in its starting config

    def generate_state_action_pairs(self):  # Oops is this full on dynamic programming bootstrapping...?
        state_action_pairs = {}
        # TODO: we should identify rotated states

        def step(board):
            for p in zip(*np.where(board.pegs == 0)):  # For each open position - TODO: verify that masked values are not used
                p_flat = board.index_flat(p)
                for move in (board.valid_moves(p)):
                    if board.valid_action(p_flat, move):
                        b = deepcopy(board)     # Split computation
                        b.apply_action(p_flat, move)
                        # No need to generate this again!
                        if (bytes(board.pegs), (p_flat, move)) not in state_action_pairs:
                            state_action_pairs[(bytes(board.pegs), (p_flat, move))] = self._score_state(b)
                            #
                            # for equiv in board.rotate_state_action(*p, move):
                            #     state_action_pairs[(bytes(equiv[0]), (equiv[1], equiv[2]))] = self._score_state(b)

                            self._render(b)
                            step(b)

        step(deepcopy(self.initial_board))
        return state_action_pairs
