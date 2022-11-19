from .base_operators import *
from .scipy_operators import *
from pydnslab.grid import Grid

__all__ = ["get_operators"]


def get_operators(grid: Grid, engine: str = "scipy") -> Operators:
    if engine == "scipy":
        return ScipyOperators(grid)
    else:
        raise NotImplementedError(f"Engine {engine} not implemented")
