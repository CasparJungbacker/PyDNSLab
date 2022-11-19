from .basesolver import Solver
from .scipysolver import ScipySolver

__all__ = ["get_solver"]


def get_solver(engine: str = "scipy") -> Solver:
    if engine == "scipy":
        return ScipySolver()
    else:
        raise NotImplementedError(f"Engine {engine} not implemented")
