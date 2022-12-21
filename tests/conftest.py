import sys
import os

import pytest

from pydnslab.case_setup import case_setup
from pydnslab.grid import Grid

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))


@pytest.fixture(scope="module")
def default_case() -> dict:
    return case_setup()


@pytest.fixture(scope="module")
def grid(default_case: dict) -> Grid:
    res = default_case["res"]
    l_scale = default_case["l_scale"]
    w_scale = default_case["w_scale"]
    return Grid(res, l_scale, w_scale)
