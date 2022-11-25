import numpy as np
import os

from scipy.io import loadmat

FIELDS_PATH = os.path.join(os.path.dirname(__file__), "..", "fields")


def load_mat_file(name: str) -> np.ndarray:
    # Load the MatLab file
    mat_file = loadmat(os.path.join(FIELDS_PATH, name))
    array = mat_file[list(mat_file)[-1]]
    # Squeeze out the extra dimension
    array = array.squeeze()
    return array
