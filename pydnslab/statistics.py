import numpy as np

from pydnslab.createfields import Fields


class Statistics:
    def __init__(self, fields: Fields, case: dict) -> None:
        self._z1 = np.arange(fields.N3 / 2 - 1, dtype=np.int32)
        self._z2 = np.arange(fields.N3 / 2 - 1, fields.N3 - 2, dtype=np.int32)
        self._y1 = np.arange(fields.N1)
        self._y2 = np.arange(fields.N1)
        self._x1 = np.arange(fields.N2)
        self._x2 = np.arange(fields.N2)

        # From settings
        self.nu = case["nu"]

        # Init statistics
        self.ut1 = None
        self.ut2 = None
        self.ut1_mean = None
        self.ut2_mean = None
        self.ut_mean = None
        self.uplu1 = None
        self.uplu2 = None
        self.vplu1 = None
        self.vplu2 = None
        self.wplu1 = None
        self.wplu2 = None
        self.uplu1_mean = None
        self.uplu2_mean = None
        self.vplu1_mean = None
        self.vplu2_mean = None
        self.wplu1_mean = None
        self.wplu2_mean = None

        self.yplu1 = None
        self.yplu1_mean = None
        self.yplu2 = None
        self.yplu2_mean = None

        self.uplumean = None
        self.vplumean = None
        self.wplumean = None
        self.yplumean = None

    def update(self, fields: Fields) -> None:
        # Shear velocity at the bottom
        self.ut1 = np.sqrt((self.nu * np.abs(fields.U[0])) / fields.Z[0])

        self.ut1_mean = self.ut1.mean()

        # Shear velocity at the top
        self.ut2 = np.sqrt((self.nu * np.abs(fields.U[-1])) / fields.Z[-1])

        self.ut2_mean = self.ut2.mean()

        self.ut_mean = 0.5 * (self.ut1_mean + self.ut2_mean)

        self.uplu1 = fields.U[self._z1] / self.ut_mean

        self.uplu1_mean = self.uplu1.mean(axis=(1, 2))

        self.uplu2 = fields.U[self._z2] / self.ut_mean
        self.uplu2 = np.flip(self.uplu2, axis=0)
        self.uplu2_mean = self.uplu2.mean(axis=(1, 2))

        self.vplu1 = fields.W[self._z1] / self.ut_mean

        self.vplu1_mean = self.vplu1.mean(axis=(1, 2))

        self.vplu2 = fields.W[self._z2] / self.ut_mean * -1
        self.vplu2 = np.flip(self.vplu2, axis=0)
        self.vplu2_mean = self.vplu2.mean(axis=(1, 2))
        self.vplu2_mean = np.flip(self.vplu2_mean, axis=0)

        self.wplu1 = fields.V[self._z1] / self.ut_mean

        self.wplu1_mean = self.wplu1.mean(axis=(1, 2))

        self.wplu2 = fields.V[self._z2] / self.ut_mean
        self.wplu2 = np.flip(self.wplu2, axis=0)
        self.wplu2_mean = self.wplu2.mean(axis=(1, 2))

        self.yplu1 = fields.Z[self._z1] / self.nu * self.ut_mean
        self.yplu1_mean = self.yplu1.mean(axis=(1, 2))

        self.yplu2 = (fields.height - fields.Z[self._z2]) / self.nu * self.ut_mean
        self.yplu2 = np.flip(self.yplu2, axis=0)
        self.yplu2_mean = self.yplu2.mean(axis=(1, 2))

        self.uplumean = 0.5 * (self.uplu1_mean + self.uplu2_mean)
        self.vplumean = 0.5 * (self.vplu1_mean + self.vplu2_mean)
        self.wplumean = 0.5 * (self.wplu1_mean + self.wplu2_mean)
        self.yplumean = 0.5 * (self.yplu1_mean + self.yplu2_mean)
        self.yplumean = self.yplumean.reshape(int(fields.N3 / 2 - 1))
