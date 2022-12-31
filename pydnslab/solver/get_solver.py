import pydnslab.config as config

from .basesolver import Solver
from .scipysolver import ScipySolver
from .cupy_solver import CupySolver

__all__ = ["get_solver"]


def get_solver() -> Solver:
    if config.backend == "scipy":
        return ScipySolver
    elif config.backend == "cupy":
        return CupySolver
    else:
        raise NotImplementedError(f"Engine {config.backend} not implemented")
