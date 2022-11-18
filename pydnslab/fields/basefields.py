from abc import ABC, abstractmethod
from typing import Any


class Fields(ABC):
    @abstractmethod
    def __init__(self, dim: tuple, runmode: int, u_nom: float, u_f: float) -> None:
        pass

    @abstractmethod
    def update(self, du: Any, dv: Any, dw: Any, pnew: Any = None):
        pass

    @property
    @abstractmethod
    def pold(self):
        pass

    @pold.setter
    @abstractmethod
    def pold(self):
        pass
