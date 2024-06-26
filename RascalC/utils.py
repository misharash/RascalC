# Contains some utility functions widely used in other scripts
# Not intended for execution from command line
import numpy as np
import os


def blank_function(*args, **kwargs) -> None:
    """
    function that accepts anything and does nothing
    mostly intended for skipping optional printing
    """
    pass


def my_a2s(a, fmt='%.18e'):
    "custom array to string function"
    return ' '.join([fmt % e for e in a])


def transposed(A: np.ndarray):
    "swap last two (matrix) axes"
    return A.swapaxes(-2, -1)


def symmetrized(A: np.ndarray):
    "symmetrize a 2+D matrix over the last two axes"
    return 0.5 * (A + transposed(A))


def rmdir_if_exists_and_empty(dirname: str) -> None:
    "remove directory if it exists and is empty, otherwise do nothing"
    if os.path.isdir(dirname) and len(os.listdir(dirname)) <= 0:
        os.rmdir(dirname)