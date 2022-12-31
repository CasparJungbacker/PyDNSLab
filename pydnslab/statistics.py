import matplotlib.pyplot as plt

import pydnslab.config as config

from pydnslab.grid import Grid
from pydnslab.fields.basefields import Fields

if config.backend == "scipy":
    import numpy as np
elif config.backend == "cupy":
    import cupy as np

__all__ = ["Statistics"]


class Statistics:
    import numpy as np

    def __init__(self, grid: Grid) -> None:
        self._z1 = np.arange(grid.N3 / 2 - 1, dtype=np.int32)
        self._z2 = np.arange(grid.N3 / 2 - 1, grid.N3 - 2, dtype=np.int32)
        self._y1 = np.arange(grid.N1)
        self._y2 = np.arange(grid.N1)
        self._x1 = np.arange(grid.N2)
        self._x2 = np.arange(grid.N2)

        # From settings
        self.nu = config.nu
        self.interval = config.stat_interval

        # List of steps at which statistics are calculated
        samples = np.arange(
            self.interval, config.nsteps + self.interval, self.interval, dtype=int
        )

        stat_dim = (len(samples), len(self._z1))

        self.row = 0

        # Declare statistics
        # TODO: pre-allocate these arrays, and don't use np.vstack later on

        self.uutmean = np.zeros(stat_dim)
        self.uuplumean = np.zeros(stat_dim)
        self.vvplumean = np.zeros(stat_dim)
        self.wwplumean = np.zeros(stat_dim)
        self.yyplumean = np.zeros(stat_dim)
        self.taveuplumean = np.zeros(stat_dim)
        self.tavevplumean = np.zeros(stat_dim)
        self.tavewplumean = np.zeros(stat_dim)
        self.taveyplumean = np.zeros(stat_dim)

        self.uufluc = np.zeros(stat_dim)
        self.vvfluc = np.zeros(stat_dim)
        self.wwfluc = np.zeros(stat_dim)
        self.uuvvfluc = np.zeros(stat_dim)
        self.uuwwfluc = np.zeros(stat_dim)
        self.vvwwfluc = np.zeros(stat_dim)

        self.uplurms = np.zeros(stat_dim)
        self.vplurms = np.zeros(stat_dim)
        self.wplurms = np.zeros(stat_dim)

        self.uupluvvplumean = np.zeros(stat_dim)
        self.uupluwwplumean = np.zeros(stat_dim)
        self.vvpluwwplumean = np.zeros(stat_dim)

    def update(self, grid: Grid, fields: Fields, i: int) -> None:

        if i % self.interval == 0 and i > 0:
            self.update_fluctuations(grid, fields)
            self.row += 1

    def plot(self) -> None:
        self._plot_fluctiations()

    def update_fluctuations(self, grid: Grid, fields: Fields) -> None:
        # Shear velocity at the bottom
        ut1 = np.sqrt((self.nu * np.abs(fields.U[0])) / grid.z[1])

        ut1_mean = ut1.mean()

        # Shear velocity at the top
        ut2 = np.sqrt(
            (self.nu * np.abs(fields.U[-1])) / (grid.height - grid.z[grid.N3 - 2])
        )

        ut2_mean = ut2.mean()

        ut_mean = 0.5 * (ut1_mean + ut2_mean)

        uplu1 = fields.U[self._z1] / ut_mean

        uplu1_mean = uplu1.mean(axis=(1, 2))

        uplu2 = fields.U[self._z2] / ut_mean
        uplu2 = np.flip(uplu2, axis=0)
        uplu2_mean = uplu2.mean(axis=(1, 2))

        vplu1 = fields.W[self._z1] / ut_mean

        vplu1_mean = vplu1.mean(axis=(1, 2))

        vplu2 = fields.W[self._z2] / ut_mean * -1
        vplu2 = np.flip(vplu2, axis=0)
        vplu2_mean = vplu2.mean(axis=(1, 2))
        vplu2_mean = np.flip(vplu2_mean, axis=0)

        wplu1 = fields.V[self._z1] / ut_mean

        wplu1_mean = wplu1.mean(axis=(1, 2))

        wplu2 = fields.V[self._z2] / ut_mean
        wplu2 = np.flip(wplu2, axis=0)
        wplu2_mean = wplu2.mean(axis=(1, 2))

        yplu1_mean = np.array(grid.z[1 : int(grid.N3 / 2)]) / self.nu * ut_mean

        yplu2 = (
            (grid.height - np.array(grid.z[int(grid.N3 / 2) : -1])) / self.nu * ut_mean
        )
        yplu2_mean = np.flip(yplu2, axis=0)

        uplumean = 0.5 * (uplu1_mean + uplu2_mean)
        vplumean = 0.5 * (vplu1_mean + vplu2_mean)
        wplumean = 0.5 * (wplu1_mean + wplu2_mean)
        yplumean = 0.5 * (yplu1_mean + yplu2_mean)
        yplumean = yplumean.reshape(int(grid.N3 / 2 - 1))

        self.uutmean[self.row] = ut_mean
        self.uuplumean[self.row] = uplumean
        self.vvplumean[self.row] = vplumean
        self.wwplumean[self.row] = wplumean
        self.yyplumean[self.row] = yplumean

        if self.row == 0:
            self.taveuplumean[self.row] = uplumean
            self.tavevplumean[self.row] = vplumean
            self.tavewplumean[self.row] = wplumean
            self.taveyplumean[self.row] = yplumean
        else:
            self.taveuplumean[self.row] = np.mean(
                self.uuplumean[: self.row + 1], axis=0
            )
            self.tavevplumean[self.row] = np.mean(
                self.vvplumean[: self.row + 1], axis=0
            )
            self.tavewplumean[self.row] = np.mean(
                self.wwplumean[: self.row + 1], axis=0
            )
            self.taveyplumean[self.row] = np.mean(
                self.yyplumean[: self.row + 1], axis=0
            )

        matuplumean = np.zeros((len(self.taveuplumean[0]), grid.N1, grid.N2))
        matvplumean = np.zeros((len(self.tavevplumean[0]), grid.N1, grid.N2))
        matwplumean = np.zeros((len(self.tavewplumean[0]), grid.N1, grid.N2))

        for nd in range(grid.N1):
            for md in range(grid.N2):
                matuplumean[:, nd, md] = self.taveuplumean[self.row]
                matvplumean[:, nd, md] = self.tavevplumean[self.row]
                matwplumean[:, nd, md] = self.tavewplumean[self.row]

        uflucsq = self._calc_sq_fluc(uplu1, uplu2, matuplumean)
        vflucsq = self._calc_sq_fluc(vplu1, vplu2, matvplumean)
        wflucsq = self._calc_sq_fluc(wplu1, wplu2, matwplumean)

        uvfluc = self._calc_cross_sq_fluc(
            uplu1, vplu1, uplu2, vplu2, matuplumean, matvplumean
        )
        uwfluc = self._calc_cross_sq_fluc(
            uplu1, wplu1, uplu2, wplu2, matuplumean, matwplumean
        )
        vwfluc = self._calc_cross_sq_fluc(
            vplu1, wplu1, vplu2, wplu2, matvplumean, matwplumean
        )

        self.uufluc[self.row] = uflucsq
        self.vvfluc[self.row] = vflucsq
        self.wwfluc[self.row] = wflucsq
        self.uuvvfluc[self.row] = uvfluc
        self.uuwwfluc[self.row] = uwfluc
        self.vvwwfluc[self.row] = vwfluc

        if self.row == 0:
            self.uplurms[self.row] = np.sqrt(uflucsq)
            self.vplurms[self.row] = np.sqrt(vflucsq)
            self.wplurms[self.row] = np.sqrt(wflucsq)
            self.uupluvvplumean[self.row] = uvfluc
            self.uupluwwplumean[self.row] = uwfluc
            self.vvpluwwplumean[self.row] = vwfluc
        else:
            self.uplurms[self.row] = np.square(
                np.mean(self.uufluc[: self.row + 1], axis=0)
            )
            self.vplurms[self.row] = np.square(
                np.mean(self.vvfluc[: self.row + 1], axis=0)
            )
            self.wplurms[self.row] = np.square(
                np.mean(self.wwfluc[: self.row + 1], axis=0)
            )
            self.uupluvvplumean[self.row] = np.mean(
                self.uuvvfluc[: self.row + 1], axis=0
            )
            self.uupluwwplumean[self.row] = np.mean(
                self.uuwwfluc[: self.row + 1], axis=0
            )
            self.vvpluwwplumean[self.row] = np.mean(
                self.vvwwfluc[: self.row + 1], axis=0
            )

    @staticmethod
    def _calc_sq_fluc(
        plu1: np.ndarray, plu2: np.ndarray, matplumean: np.ndarray
    ) -> np.ndarray:
        fluc1 = plu1 - matplumean
        fluc2 = plu2 - matplumean
        fluc1sq = np.square(fluc1)
        fluc2sq = np.square(fluc2)
        flucsq = 0.5 * (fluc1sq + fluc2sq)
        flucsq = np.mean(flucsq, axis=(1, 2))
        return flucsq

    @staticmethod
    def _calc_cross_sq_fluc(
        plu1a: np.ndarray,
        plu1b: np.ndarray,
        plu2a: np.ndarray,
        plu2b: np.ndarray,
        matplumean_a: np.ndarray,
        matplumean_b: np.ndarray,
    ) -> np.ndarray:
        fluc1 = plu1a * plu1b - matplumean_a * matplumean_b
        fluc2 = plu2a * plu2b - matplumean_a * matplumean_b
        fluc = 0.5 * (fluc1 + fluc2)
        fluc = np.mean(fluc, axis=(1, 2))
        return fluc

    def _plot_fluctiations(self) -> None:
        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()

        ax[0].semilogx(self.yyplumean[-2], self.uuplumean[-2], label="Present")
        ax[0].set_xlabel("$y^+$")
        ax[0].set_ylabel("$U^+$")
        ax[0].grid()
        ax[0].legend()
        ax[0].set_aspect("auto")

        ax[1].plot(self.yyplumean[-2], self.uplurms[-2], label="$u^+$")
        ax[1].plot(self.yyplumean[-2], self.vplurms[-2], label="$v^+$")
        ax[1].plot(self.yyplumean[-2], self.wplurms[-2], label="$w^+$")
        ax[1].set_xlabel("$y^+$")
        ax[1].set_ylabel("RMS")
        ax[1].grid()
        ax[1].legend()
        ax[1].set_aspect("auto")

        ax[2].plot(self.yyplumean[-2], self.uupluvvplumean[-2], label="$u^+v^+$")
        ax[2].set_xlabel("$y^+$")
        ax[2].set_ylabel("Mean")
        ax[2].grid()
        ax[2].legend()
        ax[2].set_aspect("auto")

        fig.delaxes(ax[-1])
        fig.tight_layout()
        fig.show()
