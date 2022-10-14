import os
import pytest
import numpy as np

from scipy.io import loadmat

from pydnslab.createfields import Fields

FIELDS_PATH = os.path.join(os.path.dirname(__file__), 'fields')

DEFAULT_CASE = dict(
    res=32,
    w_scale=0.5,
    l_scale=0.5,
    runmode=1,
    re_tau=180,
    u_nom=180*17.5*1.9e-3/0.5, # TODO: This should not be independent
    u_f=0.4
)

def load_mat_file(name: str) -> np.ndarray:
    # Load the MatLab file
    mat_file = loadmat(os.path.join(FIELDS_PATH, name))
    array = mat_file[list(mat_file)[-1]]
    # Matlab arrays are stored in Fortran order, so we need to reorder
    if array.ndim > 1:
        array = array.flatten(order='F').reshape(np.shape(array))
    return array


@pytest.fixture
def fields() -> Fields:
    return Fields(DEFAULT_CASE)


def test_num_gridpoints(fields: Fields) -> None:
    assert fields.N1 == 32
    assert fields.N2 == 32
    assert fields.N3 == 34


def test_steps(fields: Fields) -> None:
    assert fields.dx == pytest.approx(0.196349540849362)
    assert fields.dy == pytest.approx(0.098174770424681)
    assert fields.dz == pytest.approx(0.062500000000000)
    

def test_enum_matrix(fields: Fields) -> None:
    A = load_mat_file('A.mat')
    A -= 1 # Matlab is 1-indexed, Python is 0-indexed
    np.testing.assert_array_equal(fields.A, A)


def test_xyz(fields: Fields) -> None:
    


def test_cell_size(fields: Fields) -> None:
    pass


def test_XYZ(fields: Fields) -> None:
    pass


def test_indices(fields: Fields) -> None:
    pass


def test_UVW(fields: Fields) -> None:
    pass


def test_index_matrices(fields: Fields) -> None:
    pass


def test_uvw(fields: Fields) -> None:
    pass
