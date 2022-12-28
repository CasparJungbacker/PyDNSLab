import cupy as cp

from pydnslab.fields.basefields import Fields

__all__ = ["CupyFields"]


class CupyFields(Fields):
    def __init__(self, dim: tuple, runmode: int, u_nom: float, u_f: float) -> None:
        self.U = cp.zeros(dim)
        self.V = cp.zeros(dim)
        self.W = cp.zeros(dim)

        if runmode == 0:
            self.U[:, :, :] = u_nom
        elif runmode == 1:
            uf1 = u_f * (cp.random.random_sample(dim) - 0.5)
            uf2 = u_f * (cp.random.random_sample(dim) - 0.5)
            uf3 = u_f * (cp.random.random_sample(dim) - 0.5)
            self.U = self.U + uf1 * cp.amax(uf1)
            self.V = self.V + uf2 * cp.amax(uf2)
            self.W = self.W + uf3 * cp.amax(uf3)
        elif runmode == 2:
            raise NotImplementedError

        self.u = self.U.flatten()
        self.v = self.V.flatten()
        self.w = self.W.flatten()

        self._pold = cp.zeros(dim[0] * dim[1] * dim[2])

    def update(
        self, du: cp.ndarray, dv: cp.ndarray, dw: cp.ndarray, pnew: cp.ndarray = None
    ) -> None:
        self.u = self.u + du
        self.v = self.v + dv
        self.w = self.w + dw

        self.U = cp.reshape(self.u, cp.shape(self.U))
        self.V = cp.reshape(self.v, cp.shape(self.V))
        self.W = cp.reshape(self.w, cp.shape(self.W))

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
        if cp.shape(pnew) != cp.shape(self._pold):
            raise ValueError(
                f"Shape {cp.shape(pnew)} of new pressure does not match shape {cp.shape(self._pold)} of old pressure"
            )

        self._pold = pnew
