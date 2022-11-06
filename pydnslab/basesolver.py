import numpy as np

from abc import ABC, abstractmethod

from pydnslab.createfields import Fields
from pydnslab.base_operators import Operators


class Solver(ABC):
    @staticmethod
    @abstractmethod
    def projections(fields: Fields, operators: Operators) -> None:
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
        s: int,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        dt: float,
        nu: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> None:
        pass
