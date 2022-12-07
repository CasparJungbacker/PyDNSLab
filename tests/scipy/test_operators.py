import os
import pytest
import numpy as np
import scipy as sp

from utils import load_operators

from pydnslab.operators.scipy_operators import ScipyOperators
from pydnslab.grid import Grid
from pydnslab.case_setup import case_setup

OPERATORS_PATH = os.path.join(os.path.dirname(__file__), "operators")

DEFAULT_CASE = case_setup()


@pytest.fixture
def grid() -> Grid:
    res = DEFAULT_CASE["res"]
    l_scale = DEFAULT_CASE["l_scale"]
    w_scale = DEFAULT_CASE["w_scale"]
    return Grid(res, l_scale, w_scale)


@pytest.mark.parametrize("ind, mat", [(2, "Dx.txt"), (1, "Dy.txt"), (3, "Dz.txt")])
def test_differentiate_1(grid: Grid, ind: int, mat: str) -> None:
    rows, cols, data = load_operators(mat)
    M = ScipyOperators.differentiate_1(grid, ind)
    i, j, v = sp.sparse.find(M)
    np.testing.assert_array_equal(rows, i)
    np.testing.assert_array_equal(cols, j)
    np.testing.assert_almost_equal(data, v)


@pytest.mark.parametrize("ind, mat", [(2, "Dxp.txt"), (1, "Dyp.txt"), (3, "Dzp.txt")])
def test_differentiate_1p(grid: Grid, ind: int, mat: str) -> None:
    rows, cols, data = load_operators(mat)
    M = ScipyOperators.differentiate_1p(grid, ind)
    i, j, v = sp.sparse.find(M)
    np.testing.assert_array_equal(rows, i)
    np.testing.assert_array_equal(cols, j)
    np.testing.assert_almost_equal(data, v)


@pytest.mark.parametrize("ind, mat", [(2, "Dxx.txt"), (1, "Dyy.txt"), (3, "Dzz.txt")])
def test_differentiate_2(grid: Grid, ind: int, mat: str) -> None:
    rows, cols, data = load_operators(mat)
    M = ScipyOperators.differentiate_2(grid, ind)
    i, j, v = sp.sparse.find(M)
    np.testing.assert_array_equal(rows, i)
    np.testing.assert_array_equal(cols, j)
    np.testing.assert_almost_equal(data, v)


def test_poisson(grid: Grid):
    rows, cols, data = load_operators("M.txt")
    M = ScipyOperators.poisson_matrix(grid)
    i, j, v = sp.sparse.find(M)
    np.testing.assert_array_equal(rows, i)
    np.testing.assert_array_equal(cols, j)
    np.testing.assert_almost_equal(data, v)
