import pytest

from pydnslab.createfields import Fields

DEFAULT_CASE = dict(
    res=32,
    w_scale=0.5,
    l_scale=0.5,
    runmode=1,
    re_tau=180,
    u_nom=180*17.5*1.9e-3/0.5, # TODO: This should not be independent
    u_f=0.4
)

@pytest.fixture
def fields() -> Fields:
    return Fields(DEFAULT_CASE)

def test_num_gridpoints(fields: Fields) -> None:
    assert fields.N1 == 32
    assert fields.N2 == 32
    assert fields.N3 == 34

    
def test_enum_matrix(fields: Fields) -> None:
    pass


def test_xyz(fields: Fields) -> None:
    pass


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
