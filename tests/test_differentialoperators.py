import os
import pytest
import numpy as np

from scipy.io import loadmat

from pydnslab.createfields import Fields
from pydnslab.differentialoperators import Operators

FIELDS_PATH = os.path.join(os.path.dirname(__file__), "fields")

DEFAULT_CASE = dict(
    res=32,
    w_scale=0.5,
    l_scale=0.5,
    runmode=1,
    re_tau=180,
    u_nom=180 * 17.5 * 1.9e-3 / 0.5,  # TODO: This should not be independent
    u_f=0.4,
)


def load_mat_file(name: str) -> np.ndarray:
    # Load the MatLab file
    mat_file = loadmat(os.path.join(FIELDS_PATH, name))
    array = mat_file[list(mat_file)[-1]]
    # Matlab arrays are stored in Fortran order, so we need to reorder
    # array = array.ravel(order='F').reshape(np.shape(array))
    array = array.squeeze()
    return array


@pytest.fixture
def operators() -> Operators:
    fields = Fields(DEFAULT_CASE)
    return Operators(fields)


@pytest.mark.parametrize(
    "attr, mat", [("Dx", "Dx.mat"), ("Dy", "Dy.mat"), ("Dz", "Dz.mat")]
)
def test_first_order_derivatives(operators: Operators) -> None:
    pass
