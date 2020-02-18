
from typing import Dict
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

from env import Environment
from peg_solitaire.board import *
from peg_solitaire.board_drawer import BoardDrawer


class PegSolitaireEnvironment(Environment):

    def plot(self, episode):
        pass

    @property
    def state_key(self):
        return bytes(self.board.pegs)

    board_drawer: BoardDrawer
    board: Board
    _initial_board: Board  # A copy of the board in its initial state (pre-training)

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
        self.board.apply_action(self.board.index_2d(flat_index), move)

        if self.should_render:
            self.render()

        # With sparse we'd want to avoid this...
        # though for now, these are some candidate reward functions:
        # x = remaining pegs
        # https://www.desmos.com/calculator/opziinwhac
        # IDEA: compute divergence from center of mass and use that as a factor of the reward,
        #       ideally helping the agent keeping pegs close together :thinking:

        x = self.board.pegs_remaining
        n = self.board.hole_count
        # reward = ((x-(n-1)) ** 4) / ((n-2) ** 4)
        reward = 1-(x-1)/(n-1)
        # reward = ((x-(n-1)) ** 2) / ((n-2) ** 2)
        can_do_more = len(self.actions()) > 0

        return reward, not can_do_more

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

        matplotlib.use(backend="TkAgg")  # TODO: move this somewhere better?
        self.fig, self.ax = plt.subplots()

        self._initial_board = deepcopy(self.board)
        # More?
        pass

    def set_state(self, state: np.ndarray):
        self.board.pegs = state
        # More?
        pass

    def reset(self):
        self.board = deepcopy(self._initial_board)
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
        return (board.full_count - board.pegs_remaining - 1) / board.full_count

    def user_modify(self):
        self.render()
        wait_for_enter_key = True

        def on_click(event):
            print(self.board_drawer.index_from_screen_space(self.board, event.x, event.y))
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))

        cid_mousepress = self.fig.canvas.mpl_connect('button_press_event', on_click)

        def on_key_press(event: KeyEvent):
            nonlocal wait_for_enter_key
            print(f'Key pressed: {event.key}')
            if event.key == "r":
                np.random.shuffle(self.board._unmasked_pegs)
                print(self.board)
                self.render()
            if event.key == "enter":
                wait_for_enter_key = False
        cid_keypress = self.fig.canvas.mpl_connect('key_press_event', on_key_press)

        while wait_for_enter_key:
            plt.waitforbuttonpress(timeout=100)

        self.fig.canvas.mpl_disconnect(cid_mousepress)
        self.fig.canvas.mpl_disconnect(cid_keypress)

        self._initial_board = deepcopy(self.board)  # Store aside the board in its starting config

    def actions(self):
        return self._actions(self.board)

    @staticmethod
    def _actions(board):
        if board.pegs_remaining == 1:
            return []
        valid_actions = []
        for p in zip(*np.where(board.pegs == 0)):  # For each open position
            for move in (board.valid_moves(p)):
                if board.valid_action(p, move):
                    p_flat = board.index_flat(p)
                    valid_actions.append((p_flat, move))
        return valid_actions

    def generate_state_action_pairs(self):  # Oops is this full on dynamic programming bootstrapping...?
        state_action_pairs = {}
        state_values = {}
        # TODO: we should identify rotated states
        # Note: we can extract states for the critic

        def step(board):
            state_values[bytes(board.pegs)] = self._score_state(board)
            for p in zip(*np.where(board.pegs == 0)):  # For each open position - TODO: verify that masked values are not used

                for move in (board.valid_moves(p)):
                    if board.valid_action(p, move):
                        b = deepcopy(board)     # Split computation
                        b.apply_action(p, move)

                        p_flat = board.index_flat(p)

                        # No need to generate this again!
                        if (bytes(board.pegs), (p_flat, move)) not in state_action_pairs:
                            state_action_pairs[(bytes(board.pegs), (p_flat, move))] = self._score_state(b)

                            # for equiv in board.rotate_state_action(*p, move):
                            #     state_action_pairs[(bytes(equiv[0]), (equiv[1], equiv[2]))] = self._score_state(b)

                            self._render(b)
                            step(b)

        step(deepcopy(self._initial_board))
        return state_action_pairs, state_values
