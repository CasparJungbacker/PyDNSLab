import numpy as np

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

        # Init statistics
        self.uplumean = None
        self.vplumean = None
        self.wplumean = None
        self.yplumean = None

    def update(self, fields: Fields) -> None:
        # Shear velocity at the bottom
        ut1 = np.sqrt((self.nu * np.abs(fields.U[0])) / fields.Z[0])

        ut1_mean = ut1.mean()

        # Shear velocity at the top
        ut2 = np.sqrt((self.nu * np.abs(fields.U[-1])) / fields.Z[-1])

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

        yplu1 = fields.Z[self._z1] / self.nu * ut_mean
        yplu1_mean = yplu1.mean(axis=(1, 2))

        yplu2 = (fields.height - fields.Z[self._z2]) / self.nu * ut_mean
        yplu2 = np.flip(yplu2, axis=0)
        yplu2_mean = yplu2.mean(axis=(1, 2))

        self.uplumean = 0.5 * (uplu1_mean + uplu2_mean)
        self.vplumean = 0.5 * (vplu1_mean + vplu2_mean)
        self.wplumean = 0.5 * (wplu1_mean + wplu2_mean)
        self.yplumean = 0.5 * (yplu1_mean + yplu2_mean)
        self.yplumean = self.yplumean.reshape(int(fields.N3 / 2 - 1))
