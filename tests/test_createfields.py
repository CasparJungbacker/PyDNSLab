import pytest

from pydnslab.createfields import Fields


@pytest.fixture
def fields() -> Fields:
    res = 32
    w_scale = 1/2
    l_scale = 1/2
    return Fields(res, w_scale, l_scale)


def test_num_gridpoints(fields: Fields) -> None:
    assert fields.N1 == 32
    assert fields.N2 == 32
    assert fields.N3 == 34