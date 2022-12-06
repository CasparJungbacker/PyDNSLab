import numpy as np
import scipy as sp
import os


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def load_mat_file(name: str) -> np.ndarray:
    # Load the MatLab file
    mat_file = sp.io.loadmat(os.path.join(DATA_PATH, name))
    array = mat_file[list(mat_file)[-1]]
    # Squeeze out the extra dimension
    array = array.squeeze()
    return array


def load_operators(name: str) -> tuple[np.ndarray, ...]:
    arr = np.genfromtxt(os.path.join(DATA_PATH, name))
    rows = arr[:, 0] - 1
    cols = arr[:, 1] - 1
    data = arr[:, 2]
    return rows, cols, data
