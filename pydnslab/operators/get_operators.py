import pydnslab.config as config

from .base_operators import *
from .scipy_operators import *
from .cupy_operators import *
from pydnslab.grid import Grid

__all__ = ["get_operators"]


def get_operators(grid: Grid) -> Operators:
    if config.backend == "scipy":
        return ScipyOperators(grid)
    elif config.backend == "cupy":
        return CupyOperators(grid)
    # TODO: checking for invalid backends should be done by the config module
    # (maybe use ConfigParser even)
    else:
        raise NotImplementedError(f"Backend {config.backend} not implemented")
