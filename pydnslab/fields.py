import pydnslab.config as config

if config.backend == "cupy":
    try:
        import cupy as xp
    except ImportError:
        import numpy as xp
else:
    import numpy as xp

from pydnslab.grid import Grid


class Fields:
    def __init__(self, grid: Grid) -> None:
        dim = grid.griddim

        self.U = xp.zeros(dim)
        self.V = xp.zeros(dim)
        self.W = xp.zeros(dim)

        if config.runmode == 0:
            u_nom = (config.re_tau * 17.5 * config.nu) / (config.heigth / 2)
            self.U[:, :, :] = u_nom
        elif config.runmode == 1:
            uf1 = config.u_f * (xp.random.random_sample(dim) - 0.5)
            uf2 = config.u_f * (xp.random.random_sample(dim) - 0.5)
            uf3 = config.u_f * (xp.random.random_sample(dim) - 0.5)
            self.U = self.U + uf1 * xp.amax(uf1)
            self.V = self.V + uf2 * xp.amax(uf2)
            self.W = self.W + uf3 * xp.amax(uf3)
        elif config.runmode == 2:
            raise NotImplementedError

        self.u = self.U.flatten()
        self.v = self.V.flatten()
        self.w = self.W.flatten()

        self._pold = xp.zeros(dim[0] * dim[1] * dim[2])

    def update(
        self, du: xp.ndarray, dv: xp.ndarray, dw: xp.ndarray, pnew: xp.ndarray = None
    ) -> None:
        self.u = self.u + du
        self.v = self.v + dv
        self.w = self.w + dw

        self.U = xp.reshape(self.u, xp.shape(self.U))
        self.V = xp.reshape(self.v, xp.shape(self.V))
        self.W = xp.reshape(self.w, xp.shape(self.W))

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
        if xp.shape(pnew) != xp.shape(self._pold):
            raise ValueError(
                f"Shape {xp.shape(pnew)} of new pressure does not match shape {xp.shape(self._pold)} of old pressure"
            )

        self._pold = pnew
