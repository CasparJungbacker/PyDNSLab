import numpy as np

from abc import ABC, abstractmethod

from pydnslab.fields.basefields import Fields
from pydnslab.operators import Operators
from pydnslab.grid import Grid

__all__ = ["Solver"]


class Solver(ABC):
    @staticmethod
    @abstractmethod
    def projection(fields: Fields, operators: Operators) -> Fields:
        pass

    @staticmethod
    @abstractmethod
    def adjust_timestep(fields: Fields, dt: float, co_target: float) -> float:
        pass

    @abstractmethod
    def timestep(
        self,
        fields: Fields,
        operators: Operators,
        grid: Grid,
        s: int,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        dt: float,
        nu: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> Fields:
        pass
