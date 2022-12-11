import numpy as np
import pandas as pd

from pydnslab.grid import Grid
from pydnslab.fields.basefields import Fields

__all__ = ["Statistics"]


class Statistics:
    def __init__(self, grid: Grid, case: dict) -> None:
        self._z1 = np.arange(grid.N3 / 2 - 1, dtype=np.int32)
        self._z2 = np.arange(grid.N3 / 2 - 1, grid.N3 - 2, dtype=np.int32)
        self._y1 = np.arange(grid.N1)
        self._y2 = np.arange(grid.N1)
        self._x1 = np.arange(grid.N2)
        self._x2 = np.arange(grid.N2)

        # From settings
        self.nu = case["nu"]
        self.interval = case["interval"]

        # List of steps at which statistics are calculated
        samples = np.arange(
            self.interval, case["nsteps"] + self.interval, self.interval, dtype=int
        )

        # Declare statistics
        # TODO: pre-allocate these arrays, and don't use np.vstack later on
        self.uplumean = None
        self.vplumean = None
        self.wplumean = None
        self.yplumean = None
        self.ut_mean = None

        self.uutmean = None
        self.uuplumean = None
        self.vvplumean = None
        self.wwplumean = None
        self.yyplumean = None
        self.taveuplumean = None
        self.tavevplumean = None
        self.tavewplumean = None
        self.taveyplumean = None

        self.uufluc = None
        self.vvfluc = None
        self.wwfluc = None
        self.uuvvfluc = None
        self.uuwwfluc = None
        self.vvwwfluc = None

        self.uplurms = None
        self.vplurms = None
        self.wplurms = None

        self.uupluvvplumean = None
        self.uupluwwplumean = None
        self.vvpluwwplumean = None

    def update(self, grid: Grid, fields: Fields, i: int) -> None:
        if i % self.interval == 0:
            self.update_fluctuations(grid, fields)

    def update_fluctuations(self, grid: Grid, fields: Fields) -> None:
        # Shear velocity at the bottom
        ut1 = np.sqrt((self.nu * np.abs(fields.U[0])) / grid.z[1])

        ut1_mean = ut1.mean()

        # Shear velocity at the top
        ut2 = np.sqrt(
            (self.nu * np.abs(fields.U[-1])) / (grid.height - grid.z[grid.N3 - 2])
        )

        ut2_mean = ut2.mean()

        self.ut_mean = 0.5 * (ut1_mean + ut2_mean)

        uplu1 = fields.U[self._z1] / self.ut_mean

        uplu1_mean = uplu1.mean(axis=(1, 2))

        uplu2 = fields.U[self._z2] / self.ut_mean
        uplu2 = np.flip(uplu2, axis=0)
        uplu2_mean = uplu2.mean(axis=(1, 2))

        vplu1 = fields.W[self._z1] / self.ut_mean

        vplu1_mean = vplu1.mean(axis=(1, 2))

        vplu2 = fields.W[self._z2] / self.ut_mean * -1
        vplu2 = np.flip(vplu2, axis=0)
        vplu2_mean = vplu2.mean(axis=(1, 2))
        vplu2_mean = np.flip(vplu2_mean, axis=0)

        wplu1 = fields.V[self._z1] / self.ut_mean

        wplu1_mean = wplu1.mean(axis=(1, 2))

        wplu2 = fields.V[self._z2] / self.ut_mean
        wplu2 = np.flip(wplu2, axis=0)
        wplu2_mean = wplu2.mean(axis=(1, 2))

        yplu1_mean = grid.z[1 : int(grid.N3 / 2)] / self.nu * self.ut_mean

        yplu2 = (grid.height - grid.z[int(grid.N3 / 2) : -1]) / self.nu * self.ut_mean
        yplu2_mean = np.flip(yplu2, axis=0)

        self.uplumean = 0.5 * (uplu1_mean + uplu2_mean)
        self.vplumean = 0.5 * (vplu1_mean + vplu2_mean)
        self.wplumean = 0.5 * (wplu1_mean + wplu2_mean)
        self.yplumean = 0.5 * (yplu1_mean + yplu2_mean)
        self.yplumean = self.yplumean.reshape(int(grid.N3 / 2 - 1))

        if self.uplumean is None:
            self.uutmean = self.ut_mean
            self.uuplumean = self.uplumean
            self.vvplumean = self.vplumean
            self.wwplumean = self.wplumean
            self.yyplumean = self.yplumean
            self.taveuplumean = self.uplumean
            self.tavevplumean = self.vplumean
            self.tavewplumean = self.wplumean
            self.taveyplumean = self.yplumean
        else:
            self.uutmean = np.vstack((self.uutmean, self.ut_mean))
            self.uuplumean = np.vstack((self.uuplumean, self.uplumean))
            self.vvplumean = np.vstack((self.vvplumean, self.vplumean))
            self.wwplumean = np.vstack((self.wwplumean, self.wplumean))
            self.yyplumean = np.vstack((self.yyplumean, self.yplumean))
            self.taveuplumean = np.mean(self.uuplumean, axis=0)
            self.tavevplumean = np.mean(self.vvplumean, axis=0)
            self.tavewplumean = np.mean(self.wwplumean, axis=0)
            self.taveyplumean = np.mean(self.yyplumean, axis=0)

        matuplumean = np.zeros((len(self.taveuplumean), grid.N1, grid.N2))
        matvplumean = np.zeros((len(self.tavevplumean), grid.N1, grid.N2))
        matwplumean = np.zeros((len(self.tavewplumean), grid.N1, grid.N2))

        for nd in grid.N1:
            for md in grid.N2:
                matuplumean[:, nd, md] = self.taveuplumean
                matvplumean[:, nd, md] = self.tavevplumean
                matwplumean[:, nd, md] = self.tavewplumean

        uflucsq = self.calc_sq_fluc(uplu1, uplu2, matuplumean)
        vflucsq = self.calc_sq_fluc(vplu1, vplu2, matvplumean)
        wflucsq = self.calc_sq_fluc(wplu1, wplu2, matwplumean)

        uvfluc = self.calc_cross_sq_fluc(
            uplu1, vplu1, uplu2, vplu2, matuplumean, matvplumean
        )
        uwfluc = self.calc_cross_sq_fluc(
            uplu1, wplu1, uplu2, wplu2, matuplumean, matwplumean
        )
        vwfluc = self.calc_cross_sq_fluc(
            vplu1, wplu1, vplu2, wplu2, matvplumean, matwplumean
        )

        if self.uufluc is None:
            self.uufluc = uflucsq
            self.vvfluc = vflucsq
            self.wwfluc = wflucsq
            self.uplurms = np.sqrt(uflucsq)
            self.vplurms = np.sqrt(vflucsq)
            self.wplurms = np.sqrt(wflucsq)

            self.uuvvfluc = uvfluc
            self.uuwwfluc = uwfluc
            self.vvwwfluc = vwfluc
            self.uupluvvplumean = uvfluc
            self.uupluwwplumean = uwfluc
            self.vvpluwwplumean = vwfluc
        else:
            self.uufluc = np.vstack((self.uufluc, uflucsq))
            self.vvfluc = np.vstack((self.vvfluc, vflucsq))
            self.wwfluc = np.vstack((self.wwfluc, wflucsq))
            self.uplurms = self.calc_plurms(self.uufluc)
            self.vplurms = self.calc_plurms(self.vvfluc)
            self.wplurms = self.calc_plurms(self.wwfluc)

            self.uuvvfluc = np.vstack((self.uuvvfluc, uvfluc))
            self.uuwwfluc = np.vstack((self.uuwwfluc, uwfluc))
            self.vvwwfluc = np.vstack((self.vvwwfluc, vwfluc))

            self.uupluvvplumean = np.mean(self.uuvvfluc, axis=0)
            self.uupluwwplumean = np.mean(self.uuwwfluc, axis=0)
            self.vvpluwwplumean = np.mean(self.vvwwfluc, axis=0)

    @staticmethod
    def calc_sq_fluc(
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
    def calc_cross_sq_fluc(
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

    @staticmethod
    def calc_plurms(fluc):
        plurms = np.mean(fluc, axis=0)
        plurms = np.square(plurms)
        return plurms
