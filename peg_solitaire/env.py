import math
from typing import Dict
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

from env import Environment
from peg_solitaire.board import *
from peg_solitaire.board_drawer import BoardDrawer


class PegSolitaireEnvironment(Environment):

    @property
    def state_key(self):
        return self.board.to_bytes()

    board_drawer: BoardDrawer
    board: Board
    _initial_board: Board  # A copy of the board in its initial state (pre-training)
    _frame_delay: float
    should_render: bool

    def _render(self, board):
        self.axis.imshow(self.board_drawer.draw(board))
        plt.pause(self._frame_delay)

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

        reward = self.score_state()

        can_do_more = len(self.actions()) > 0

        return reward, not can_do_more

    def has_won(self):
        return self.board.pegs_remaining == 1

    @property
    def pegs_remaining(self) -> int:
        return self.board.pegs_remaining

    # NOTE: we want to allow creation before setup, hence this is not in __init__
    # Though if need be we may put it there later... :thinking:
    def setup(self, config: Dict):
        env_config = config['env']
        visual_config = config['visual']

        if env_config['type'] == 'triangle':
            self.board = TriangleBoard(env_config['size'])
        else:
            self.board = DiamondBoard(env_config['size'])

        board_drawer_args = {key: visual_config[key] for key in BoardDrawer.__init__.__code__.co_varnames[1:]}
        self.board_drawer = BoardDrawer(**board_drawer_args)
        self._frame_delay = visual_config['frame_delay']
        self._initial_board = deepcopy(self.board)  # Store away the original configuration of the board

        self.axis = plt.gcf().axes[0]


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
        return self._score_state(self.board)

    @staticmethod
    def _score_state(board: Board):
        """
        Naive scoring based on board fullness
        :return: a normalized score
        """

        # With sparse we'd want to avoid this...
        # though for now, these are some candidate reward functions:
        # x = remaining pegs
        # https://www.desmos.com/calculator/opziinwhac
        # IDEA: compute divergence from center of mass and use that as a factor of the reward,
        #       ideally helping the agent keeping pegs close together :thinking:
        #
        def smoothstep(x, min, max):
            if x < min: return min
            elif x > max: return max
            return (max-min) * x * x * x * (x * (x * 6 - 15) + 10)

        x0 = board.pegs_remaining
        x1 = len(PegSolitaireEnvironment._actions(board))
        n = board.hole_count
        # reward = ((x-(n-1)) ** 4) / ((n-2) ** 4)  # QUAD-SCALE REWARD [0, 1]
        # reward = 1-(x0-1)/(n-2)  # LINEAR REWARD [0, 1]

        # Locate center
        c = (0, 0)
        for y in range(len(board.pegs)):
            for x in range(len(board.pegs[0])):
                if board.pegs[y, x]:
                    c = c[0] + x, c[1] + y

        c = (c[0]/x0, c[1]/x0)  # divide positions to find center

        total_dist = 0
        for y in range(len(board.pegs)):
            for x in range(len(board.pegs[0])):
                if board.pegs[y, x]:
                    x_ = (x - c[0]) ** 2
                    y_ = (y - c[1]) ** 2
                    total_dist += math.sqrt(x_ + y_)
        # for board.center

        # p and curiosity are strongly linked
        #p = 4  # NOTE: THIS MUST RESULT IN A VALID FUNCTION - not all p-s give working functions
        #reward = abs(2*((x-(n-1)) ** p)) / ((n-2) ** p) - 1  # p-POWERED REWARD [-1, 1]

        lin_reward = (2 * (1 - x0) / (n - 2)) + 1  # LINEAR REWARD [-1, 1]
        # reward = lin_reward + PegSolitaireEnvironment._countDistinctIslands(board) / n #
        # g = smoothstep(x1*x1/n, 0, 1)

        return lin_reward + x0 / (total_dist + 0.1)



    # Function to perform dfs of the input grid
    @staticmethod
    def _dfs(moves, pegs, x0, y0, i, j, v):
        rows = len(pegs)
        cols = len(pegs[0])
        if i < 0 or i >= rows or i < 0 or j >= cols or pegs[i][j] <= 0:
            return
        # marking the visited element as -1
        pegs[i][j] *= -1

        # computing coordinates with x0, y0 as base
        v.append([i - x0, j - y0])

        # repeat dfs for neighbors
        for dir in moves:
            PegSolitaireEnvironment._dfs(moves, pegs, x0, y0, i + dir[0], j + dir[1], v)


            # Main function that returns distinct count of islands in


    @staticmethod
    def _countDistinctIslands(board):
        grid = np.copy(board.pegs).astype('float32')
        coordinates = []

        for i in range(len(grid)):
            for j in range(len(grid[0])):

                # If a cell is not 1
                # no need to dfs
                if not grid[i][j]:
                    continue

                # to hold coordinates
                # of this island
                v = []
                PegSolitaireEnvironment._dfs(board.moves, grid, i, j, i, j, v)

                # insert the coordinates for
                # this island to set
                if len(v):
                    coordinates.append(v)

        return len(coordinates)

        # Driver code

    def _render_selection(self, selection: (int, int)):

        self.board_drawer.draw(self.board)
        self.axis.imshow(self.board_drawer.draw_selection(self.board, selection))
        plt.pause(self._frame_delay)

    def user_modify(self):
        wait_for_enter_key = True

        if issubclass(type(self.board), TriangleBoard):
            key_mapping = {
                '7': (-1, -1),
                '4': (0, -1),
                '1': (1, 0),
                '9': (-1, 0),
                '6': (0, 1),
                '3': (1, 1),
            }
        else:
            key_mapping = {
                '7': (-1, 0),
                '4': (-1, -1),
                '1': (0, -1),
                '9': (0, 1),
                '6': (1, 1),
                '3': (1, 0),
            }

        selection_peg = self.board.center
        self._render_selection(selection_peg)

        def on_key_press(event: KeyEvent):
            nonlocal wait_for_enter_key
            nonlocal selection_peg
            nonlocal key_mapping
            print(f'Key pressed: {event.key}')
            if event.key in key_mapping:
                m = key_mapping[event.key]
                print(selection_peg)
                p = selection_peg[0]+m[0], selection_peg[1]+m[1]
                print(m)
                print(p)
                if 0 <= p[0] < self.board.size and 0 <= p[1] < self.board.size \
                        and not ma.is_masked(self.board.pegs[p]):
                    selection_peg = p
                    self._render_selection(selection_peg)
            if event.key == "5":
                self.board.pegs[selection_peg] = not self.board.pegs[selection_peg]
                self._render_selection(selection_peg)
            if event.key == "p":
                print(self._countDistinctIslands(self.board))
                print(self._score_state(self.board))
            if event.key == "enter":
                wait_for_enter_key = False

        fig = plt.gcf()
        cid_keypress = fig.canvas.mpl_connect('key_press_event', on_key_press)

        while wait_for_enter_key:
            plt.waitforbuttonpress(timeout=100)

        fig.canvas.mpl_disconnect(cid_keypress)

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
        # Note: we can extract states for the critic
        # we can identify rotated states

        def step(board):
            state_values[bytes(board.pegs)] = self._score_state(board)
            for p in zip(*np.where(board.pegs == 0)):  # For each open position

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
