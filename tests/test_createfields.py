from json import load
import os
import pytest
import numpy as np

from scipy.io import loadmat

from pydnslab.createfields import Fields

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
    A = load_mat_file("A.mat")
    A -= 1  # Matlab is 1-indexed, Python is 0-indexed
    A = A.flatten(order="F").reshape(np.shape(A.T))
    np.testing.assert_array_equal(fields.A, A)


def test_xyz(fields: Fields) -> None:
    x = load_mat_file("x.mat")
    y = load_mat_file("y.mat")
    z = load_mat_file("z.mat")
    np.testing.assert_array_almost_equal(fields.x, x)
    np.testing.assert_array_almost_equal(fields.y, y)
    np.testing.assert_array_almost_equal(fields.z, z)


def test_cell_size(fields: Fields) -> None:
    FX = load_mat_file("FX.mat")
    FY = load_mat_file("FY.mat")
    FZ = load_mat_file("FZ_.mat")
    np.testing.assert_array_almost_equal(fields.FX, FX.T)
    np.testing.assert_array_almost_equal(fields.FY, FY.T)
    np.testing.assert_array_almost_equal(fields.FZ, FZ.T)


def test_XYZ(fields: Fields) -> None:
    X = load_mat_file("X_.mat")
    Y = load_mat_file("Y_.mat")
    Z = load_mat_file("Z_.mat")
    np.testing.assert_almost_equal(fields.X, X)
    np.testing.assert_almost_equal(fields.Y, Y)
    np.testing.assert_almost_equal(fields.Z, Z)


def test_inx(fields: Fields) -> None:
    inx = load_mat_file("inx.mat")
    inx = inx - 1
    np.testing.assert_almost_equal(fields.inx, inx)


def test_iny(fields: Fields) -> None:
    iny = load_mat_file("iny.mat")
    iny = iny - 1
    np.testing.assert_almost_equal(fields.iny, iny)


def test_inz(fields: Fields) -> None:
    inz = load_mat_file("inz.mat")
    inz = inz - 1
    np.testing.assert_almost_equal(fields.inz, inz)


@pytest.mark.parametrize(
    "attr, mat",
    [
        ("north", "north.mat"),
        ("south", "south.mat"),
        ("east", "east.mat"),
        ("west", "west.mat"),
        ("air", "air.mat"),
        ("ground", "ground.mat"),
    ],
)
def test_directional_indices(fields: Fields, attr: str, mat: str) -> None:
    arr = load_mat_file(mat) - 1
    np.testing.assert_array_equal(getattr(fields, attr), arr)


def test_UVW(fields: Fields) -> None:
    pass


def test_A0(fields: Fields) -> None:
    A0 = load_mat_file("A0.mat")


@pytest.mark.parametrize(
    "attr, mat",
    [
        ("AN", "AN.mat"),
        ("AS", "AS.mat"),
        ("AE", "AE.mat"),
        ("AW", "AW.mat"),
        ("AA", "AA.mat"),
        ("AG", "AG.mat"),
    ],
)
def test_enum_matrices(fields: Fields, attr: str, mat: str) -> None:
    mat_field = load_mat_file(mat).flatten(order="F") - 1
    np.testing.assert_array_equal(getattr(fields, attr), mat_field)


def test_uvw(fields: Fields) -> None:
    pass
