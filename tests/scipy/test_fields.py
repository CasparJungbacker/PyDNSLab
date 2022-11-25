import os
import pytest
import numpy as np

from utils import load_mat_file

from pydnslab.grid import Grid
from pydnslab.case_setup import case_setup

FIELDS_PATH = os.path.join(os.path.dirname(__file__), "fields")

DEFAULT_CASE = case_setup()


@pytest.fixture
def grid() -> Grid:
    res = DEFAULT_CASE["res"]
    l_scale = DEFAULT_CASE["l_scale"]
    w_scale = DEFAULT_CASE["w_scale"]
    return Grid(res, l_scale, w_scale)


def test_num_gridpoints(grid: Grid) -> None:
    assert grid.N1 == 32
    assert grid.N2 == 32
    assert grid.N3 == 34


def test_steps(grid: Grid) -> None:
    assert grid.dx == pytest.approx(0.196349540849362)
    assert grid.dy == pytest.approx(0.098174770424681)
    assert grid.dz == pytest.approx(0.062500000000000)


def test_enum_matrix(grid: Grid) -> None:
    A = load_mat_file("A.mat")
    A -= 1  # Matlab is 1-indexed, Python is 0-indexed
    A = A.flatten(order="F").reshape(np.shape(A.T))
    np.testing.assert_array_equal(grid.A, A)


@pytest.mark.parametrize(
    "attr, mat",
    [
        ("x", "x.mat"),
        ("y", "y.mat"),
        ("z", "z.mat"),
    ],
)
def test_gridpoints(grid: Grid, attr: str, mat: str) -> None:
    arr = load_mat_file(mat)
    np.testing.assert_almost_equal(getattr(grid, attr), arr)


def test_xyz(grid: Grid) -> None:
    x = load_mat_file("x.mat")
    y = load_mat_file("y.mat")
    z = load_mat_file("z.mat")
    np.testing.assert_array_almost_equal(grid.x, x)
    np.testing.assert_array_almost_equal(grid.y, y)
    np.testing.assert_array_almost_equal(grid.z, z)


def test_cell_size(grid: Grid) -> None:
    FX = load_mat_file("FX.mat")
    FY = load_mat_file("FY.mat")
    FZ = load_mat_file("FZ_.mat")
    np.testing.assert_array_almost_equal(grid.FX, FX.T)
    np.testing.assert_array_almost_equal(grid.FY, FY.T)
    np.testing.assert_array_almost_equal(grid.FZ, FZ.T)


def test_XYZ(grid: Grid) -> None:
    X = load_mat_file("X_.mat")
    Y = load_mat_file("Y_.mat")
    Z = load_mat_file("Z_.mat")
    np.testing.assert_almost_equal(grid.X, X)
    np.testing.assert_almost_equal(grid.Y, Y)
    np.testing.assert_almost_equal(grid.Z, Z)


@pytest.mark.parametrize(
    "attr, mat",
    [
        ("inx", "inx.mat"),
        ("iny", "iny.mat"),
        ("inz", "inz.mat"),
    ],
)
def test_grid_indices(grid: Grid, attr: str, mat: str) -> None:
    arr = load_mat_file(mat) - 1
    np.testing.assert_equal(getattr(grid, attr), arr)


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
def test_directional_indices(grid: Grid, attr: str, mat: str) -> None:
    arr = load_mat_file(mat) - 1
    np.testing.assert_array_equal(getattr(grid, attr), arr)


def test_A0(grid: Grid) -> None:
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
def test_enum_matrices(grid: Grid, attr: str, mat: str) -> None:
    mat_field = load_mat_file(mat).flatten(order="F") - 1
    np.testing.assert_array_equal(getattr(grid, attr), mat_field)
