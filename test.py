import struct

import matplotlib.pyplot as plt
import argparse as arg
import numpy as np
import threading
import numpy.ma as ma


# A runnable file to test out various things

def index_flat(arr: np.ndarray, pos: (int, int)):
    """
    Get the index of the flattened board when
    :param pos: x, y
    :return: flat index
    """
    return pos[1] + pos[0] * arr.shape[1]


def index_2d(arr, i: int):
    """
        flat index to 2D index
        :param i: flat index
        :return: 2D tuple index
        """
    return int(i / arr.shape[1]), i % arr.shape[1]


#
# def dfs(x, y, visited, a, moves):
#     if not visited[x, y]:
#         visited[y, x] = True
#         for x_, y_ in moves:
#             if a[y+y_, x+x_]:
#                 dfs(y+y_, x+x_, visited, a, moves)


# Function to perform dfs of the input grid
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
        _dfs(moves, pegs, x0, y0, i + dir[0], j + dir[1], v)

        # Main function that returns distinct count of islands in


def _countDistinctIslands(board, moves):
    grid = np.copy(board).astype('float64')
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
            _dfs(moves, grid, i, j, i, j, v)

            # insert the coordinates for
            # this island to set
            coordinates.append(v)

    return len(coordinates)

    # Driver code


if __name__ == '__main__':
    moves = np.array([(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)])

    a = np.array([[1, 1, 0, 0, 0, 0, 1],
                  [1, 1, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]])

    visited = np.zeros_like(a, dtype=bool)
    print()
    c = 0
    a_ = np.pad(a, 1)

    # while True:
    #     visited
    #
    # for y,row in enumerate(a):
    #     for x, val in enumerate(row):
    #         if not visited[y, x]:
    #             dfs(x,y,visited,a_,moves)

    l = []
    l[2] = 1

    print(_countDistinctIslands(a, moves))

    dirs = moves

    grid = a

    #
    # indices = np.array([*zip(*np.where(a))])
    # print(len(indices))
    # com = sum(indices)/len(indices)
    # print(indices)
    # print(com)
    #
    # f = sum(indices[:]*1.1-com)
    # l = f[0] ** 2 + f[1] ** 2
    # print(l)

    pass
