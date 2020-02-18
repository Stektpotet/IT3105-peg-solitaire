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


if __name__ == '__main__':

    a = np.array([[1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1]])
    indices = np.array([*zip(*np.where(a))])
    print(len(indices))
    com = sum(indices)/len(indices)
    print(indices)
    print(com)

    f = sum(indices[:]*1.1-com)
    l = f[0] ** 2 + f[1] ** 2
    print(l)

    pass
