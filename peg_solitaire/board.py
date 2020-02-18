from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.ma as ma

class Board(ABC):
    pegs: np.ndarray
    size: int
    hole_count: int

    @property
    def shape(self): return self.pegs.shape

    @property
    def pegs_remaining(self): return np.sum(self.pegs)

    @property
    def full_count(self) -> int:
        return self.hole_count - 1

    @abstractmethod
    def _count_holes(self) -> int: pass

    @abstractmethod
    def to_bytes(self) -> bytes: pass

    @abstractmethod
    def rotate_state_action(self, x, y, move) -> List:
        """
        Allow rotation of a state-action pair to increase the knowledge of the agent given a SAP
        :param x:
        :param y:
        :param move:
        :return:
        """
        pass

    # TODO: Look at what is most useful; flat indices or 2d indices - which is more used? Then unify...

    def valid_moves(self, peg_index2d: (int, int)):
        move_indices = []
        # We don't need to check if the skippable is there,
        # We need to check if the one past the skippable is there, hence the ' *2 '
        for move_index, move in enumerate(self.moves):
            y_axis = peg_index2d[0] + move[0] * 2
            x_axis = peg_index2d[1] + move[1] * 2
            if 0 <= y_axis < self.shape[0] and 0 <= x_axis < self.shape[1]:
                if not ma.is_masked(self.pegs[y_axis, x_axis]):
                    move_indices.append(move_index)

        return move_indices

    def __init__(self, size: int):
        """
        Moves:
                    0---1
                    | \ | \
                    5---X---2
                      \ | \ |
                        4---3
        """
        self.size = size
        self.hole_count = self._count_holes()
        self.moves = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]
        pass

    def __repr__(self):
        return f"<Board\nshape: {self.shape}\npegs: \n{self.pegs}\n>"

    def peg(self, flat_index: int):
        return self.pegs.flat[flat_index]

    def index_flat(self, pos: (int, int)):
        """
        Get the index of the flattened board when
        :param pos: x, y
        :return: flat index
        """
        return pos[1] + pos[0] * self.shape[1]

    def index_2d(self, i: int):
        """
        flat index to 2D index
        :param i: flat index
        :return: 2D tuple index
        """
        return int(i / self.shape[1]), i % self.shape[1]

    def _peg_line(self, peg: (int, int), move_index: int):
        peg_skip_pos = (peg[0] + self.moves[move_index][0], peg[1] + self.moves[move_index][1])
        peg_jumper = (peg[0] + self.moves[move_index][0] * 2, peg[1] + self.moves[move_index][1] * 2)
        return peg_skip_pos, peg_jumper

    def valid_action(self, peg: (int, int), move_index: int):
        """
        Catch all cases of invalid actions before assuming it's a valid one
        :param peg: the index of the peg hole we want filled
        :param move_index: the index of which direction to source a peg from
        :return:
        """

        if self.pegs[peg] != 0:
            print(f"position {peg} is already filled (or it is masked)!")
            return False

        if move_index < 0 or move_index >= len(self.moves):
            print(f"Illegal move - not defined in 'board.moves'")
            return False

        peg_skip_pos, peg_jumper = self._peg_line(peg, move_index)

        if self.pegs[peg_skip_pos] == 0:
            # print(f"Illegal move - cannot skip over {peg_skip_pos}, it's not filled!")
            return False

        if self.pegs[peg_jumper] == 0:
            # print(f"Illegal move - cannot skip from {peg_jumper}, it's not filled!")
            return False

        return True

    def apply_action(self,  peg: (int, int), move_index: int):
        peg_skip_pos, peg_jumper = self._peg_line(peg, move_index)

        self.pegs[peg] = True
        self.pegs[peg_skip_pos] = False
        self.pegs[peg_jumper] = False

class TriangleBoard(Board):

    def to_bytes(self) -> bytes:
        return bytes([p for p in self.pegs.flat if not ma.is_masked(p)])

    def rotate_state_action(self, x, y, move) -> List:
        x = np.rot90(self._unmasked_pegs)
        for i in range(len(x) - 1, -1, -1):
            x[i] = np.roll(x[i], -(len(x) - 1 - i))

        s0 = ma.masked_array(x, mask=np.tri(self.size, dtype=np.uint8, k=-1).T, hard_mask=True)
        # p0 =

        s1 = np.rot90(np.flipud(self.pegs))
        p1 = (y, x)

        # TODO: Finish these equivalence generators
        return [(),
                ()]
        pass

    def _count_holes(self) -> int:
        return len(self.flat_indices)

    def __init__(self, size: int):
        self._unmasked_pegs = np.tri(size, dtype=np.uint8)
        self.pegs = ma.masked_array(self._unmasked_pegs, mask=np.tri(size, dtype=np.uint8, k=-1).T, hard_mask=True)
        print(bytes(self.pegs))
        self.indices = list(zip(*np.tril_indices_from(self.pegs)))

        self.flat_indices = [self.index_flat(i) for i in self.indices]
        print(len(self.flat_indices))

        self.pegs[int(self.shape[0] / 2), int(self.shape[1] / 2) - int(self.shape[1] / 4)] = 0
        Board.__init__(self, size)

    def __repr__(self):
        return f"<Board\nshape: {self.shape}\npegs: \n{self.pegs}\n>"


class DiamondBoard(Board):

    def to_bytes(self) -> bytes:
        return bytes(self.pegs)

    def rotate_state_action(self, x, y, move) -> List:
        rotated_position = self.shape[1] - x - 1, self.shape[0] - y - 1
        rotated_move = int(move + len(self.moves)/2) % len(self.moves)
        return [(np.flip(self.pegs), self.index_flat(rotated_position), rotated_move)]
        pass

    def _count_holes(self) -> int:
        """
        :return: The max number of pegs on the board, i.e. nr of holes - 1
        """
        return self.shape[0] * self.shape[1]

    def __init__(self, size: int):
        self.pegs = np.ones((size, size), dtype=np.uint8)
        self.pegs[int(self.shape[1] / 2), int(self.shape[0] / 2)] = 0
        Board.__init__(self, size)

    # NOTE: The following may be generalizable enough to put it into the superclass


