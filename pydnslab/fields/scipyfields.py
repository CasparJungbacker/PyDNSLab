import numpy as np

from pydnslab.fields.basefields import Fields

__all__ = ["ScipyFields"]


class ScipyFields(Fields):
    def __init__(self, dim: tuple, runmode: int, u_nom: float, u_f: float) -> None:
        # 3D velocity arrays
        self.U: np.ndarray = np.zeros(dim)
        self.V: np.ndarray = np.zeros(dim)
        self.W: np.ndarray = np.zeros(dim)

        if runmode == 0:
            self.U[:, :, :] = u_nom
        elif runmode == 1:
            UF1 = u_f * (np.random.random_sample(dim) - 0.5)
            UF2 = u_f * (np.random.random_sample(dim) - 0.5)
            UF3 = u_f * (np.random.random_sample(dim) - 0.5)
            self.U = self.U + UF1 * np.amax(UF1)
            self.V = self.V + UF2 * np.amax(UF2)
            self.W = self.W + UF3 * np.amax(UF3)
        elif runmode == 2:
            raise NotImplementedError

        # 1D velocity arrays
        self.u = self.U.flatten()
        self.v = self.V.flatten()
        self.w = self.W.flatten()

        # Pressure of previous iteration
        self._pold = np.zeros(dim[0] * dim[1] * dim[2])

    def update(
        self, du: np.ndarray, dv: np.ndarray, dw: np.ndarray, pnew: np.ndarray = None
    ) -> None:
        self.u = self.u + du
        self.v = self.v + dv
        self.w = self.w + dw

        if pnew is not None:
            self.pold = pnew

    @property
    def pold(self):
        return self._pold

    @pold.setter
    def pold(self, pnew):
        if not isinstance(pnew, type(self._pold)):
            raise TypeError(
                f"New pressure must be of type {type(self._pold)}, but is of type {type(pnew)}"
            )
        if np.shape(pnew) != np.shape(self._pold):
            raise ValueError(
                f"Shape {np.shape(pnew)} of new pressure does not match shape {np.shape(self._pold)} of old pressure"
            )

        self._pold = pnew
