import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import numpy as np

from pydnslab.grid import Grid

__all__ = ["ScipyOperators"]


class ScipyOperators:
    """Input: probably a grid object"""

    def __init__(self, grid: Grid):
        self.Dx = self.differentiate_1(grid, 2)
        self.Dy = self.differentiate_1(grid, 1)
        self.Dz = self.differentiate_1(grid, 3)

        self.Dxx = self.differentiate_2(
            grid,
            2,
        )
        self.Dyy = self.differentiate_2(
            grid,
            1,
        )
        self.Dzz = self.differentiate_2(
            grid,
            3,
        )

        self.Dxp = self.differentiate_1p(grid, 2)
        self.Dyp = self.differentiate_1p(grid, 1)
        self.Dzp = self.differentiate_1p(grid, 3)

        self.M = self.poisson_matrix(grid)

        # Preconditioner
        self.M_inv_approx = self.preconditioner(
            self.M, grid.N1 * grid.N2 * (grid.N3 - 2)
        )

    @staticmethod
    def differentiate_1(
        grid: Grid,
        index: int,
    ) -> sps.spmatrix:
        """First order derivatives"""

        FY0 = grid.FY[1:-1].flatten()
        FYN = grid.FY[1:-1, :, grid.north].flatten()
        FYS = grid.FY[1:-1, :, grid.south].flatten()

        FX0 = grid.FX[1:-1].flatten()
        FXE = grid.FX[1:-1, grid.east].flatten()
        FXW = grid.FX[1:-1, grid.west].flatten()

        FZ0 = grid.FZ[1:-1].flatten()
        FZA = grid.FZ[grid.air].flatten()
        FZG = grid.FZ[grid.ground].flatten()

        if index == 1:
            data_1 = (1 / FY0) * (FYN / (FY0 + FYN) - FYS / (FYS + FY0))
            data_2 = 1 / FY0 * FY0 / (FY0 + FYN)
            data_3 = -1 / FY0 * FY0 / (FY0 + FYS)

            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate(
                (
                    grid.A0.flatten(),
                    grid.AN[grid.A0.flatten()],
                    grid.AS[grid.A0.flatten()],
                )
            )
            data = np.concatenate((data_1, data_2, data_3))

        elif index == 2:
            data_1 = (1 / FX0) * (FXE / (FX0 + FXE) - FXW / (FXW + FX0))
            data_2 = 1 / FX0 * FX0 / (FX0 + FXE)
            data_3 = -1 / FX0 * FX0 / (FX0 + FXW)

            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate(
                (
                    grid.A0.flatten(),
                    grid.AE[grid.A0.flatten()],
                    grid.AW[grid.A0.flatten()],
                )
            )
            data = np.concatenate((data_1, data_2, data_3))

        elif index == 3:
            data_1 = (1 / FZ0) * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0))

            mask = grid.AG[grid.A0.flatten()] >= 0

            data_2 = (-1 / FZ0 * FZ0 / (FZ0 + FZG))[mask]
            data_1[~mask] = (
                1 / FZ0 * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0) + FZ0 / (FZG + FZ0))
            )[~mask]

            cols_2 = grid.AG[grid.A0.flatten()[mask]]

            rows_2 = grid.A0.flatten()[mask]

            mask = grid.AA[grid.A0.flatten()] <= int(
                grid.N1 * grid.N2 * (grid.N3 - 2) - 1
            )

            data_3 = (1 / FZ0 * FZ0 / (FZ0 + FZA))[mask]
            data_1[~mask] = (
                1 / FZ0 * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0) - FZ0 / (FZA + FZ0))
            )[~mask]

            cols_3 = grid.AA[grid.A0.flatten()[mask]]

            rows_3 = grid.A0.flatten()[mask]

            data = np.concatenate((data_1, data_2, data_3))
            rows = np.concatenate((grid.A0.flatten(), rows_2, rows_3))
            cols = np.concatenate((grid.A0.flatten(), cols_2, cols_3))

        else:
            raise ValueError(f"Invalid index: {index}")

        N = len(grid.A0.flatten())

        M = sps.csr_matrix((data, (rows, cols)), shape=(N, N))
        M.eliminate_zeros()

        return M

    @staticmethod
    def differentiate_1p(
        grid: Grid,
        index: int,
    ) -> sps.dok_matrix:
        """First order pressure derivates"""
        FY0 = grid.FY[1:-1].flatten()
        FYN = grid.FY[1:-1, :, grid.north].flatten()
        FYS = grid.FY[1:-1, :, grid.south].flatten()

        FX0 = grid.FX[1:-1].flatten()
        FXE = grid.FX[1:-1, grid.east].flatten()
        FXW = grid.FX[1:-1, grid.west].flatten()

        FZ0 = grid.FZ[1:-1].flatten()
        FZA = grid.FZ[grid.air].flatten()
        FZG = grid.FZ[grid.ground].flatten()

        if index == 1:
            data_1 = (1 / FY0) * (FYN / (FY0 + FYN) - FYS / (FYS + FY0))
            data_2 = 1 / FY0 * FY0 / (FY0 + FYN)
            data_3 = -1 / FY0 * FY0 / (FY0 + FYS)

            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate(
                (
                    grid.A0.flatten(),
                    grid.AN[grid.A0.flatten()],
                    grid.AS[grid.A0.flatten()],
                )
            )
            data = np.concatenate((data_1, data_2, data_3))

        elif index == 2:
            data_1 = (1 / FX0) * (FXE / (FX0 + FXE) - FXW / (FXW + FX0))
            data_2 = 1 / FX0 * FX0 / (FX0 + FXE)
            data_3 = -1 / FX0 * FX0 / (FX0 + FXW)

            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate(
                (
                    grid.A0.flatten(),
                    grid.AE[grid.A0.flatten()],
                    grid.AW[grid.A0.flatten()],
                )
            )
            data = np.concatenate((data_1, data_2, data_3))

        elif index == 3:
            data_1 = (1 / FZ0) * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0))

            mask = grid.AG[grid.A0.flatten()] >= 0

            data_2 = (-1 / FZ0 * (FZ0 / (FZ0 + FZG)))[mask]
            data_1[~mask] = (
                1 / FZ0 * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0) - FZ0 / (FZG + FZ0))
            )[~mask]

            cols_2 = grid.AG[mask]
            rows_2 = grid.A0.flatten()[mask]

            mask = grid.AA[grid.A0.flatten()] <= int(
                grid.N1 * grid.N2 * (grid.N3 - 2) - 1
            )

            data_3 = (1 / FZ0 * FZ0 / (FZ0 + FZA))[mask]
            data_1[~mask] = (
                1 / FZ0 * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0) + FZ0 / (FZA + FZ0))
            )[~mask]

            cols_3 = grid.AA[mask]
            rows_3 = grid.A0.flatten()[mask]

            data = np.concatenate((data_1, data_2, data_3))
            rows = np.concatenate((grid.A0.flatten(), rows_2, rows_3))
            cols = np.concatenate((grid.A0.flatten(), cols_2, cols_3))

        else:
            raise ValueError(f"Invalid index: {index}")

        N = len(grid.A0.flatten())

        M = sps.coo_matrix((data, (rows, cols)), shape=(N, N))
        M.eliminate_zeros()

        return M

    @staticmethod
    def differentiate_2(
        grid: Grid,
        index: int,
    ) -> sps.dok_matrix:
        """Second order derivatives"""

        FY0 = grid.FY[1:-1].flatten()
        FYN = grid.FY[1:-1, :, grid.north].flatten()
        FYS = grid.FY[1:-1, :, grid.south].flatten()

        FX0 = grid.FX[1:-1].flatten()
        FXE = grid.FX[1:-1, grid.east].flatten()
        FXW = grid.FX[1:-1, grid.west].flatten()

        FZ0 = grid.FZ[1:-1].flatten()
        FZA = grid.FZ[grid.air].flatten()
        FZG = grid.FZ[grid.ground].flatten()

        if index == 1:
            data_1 = -2 / (FY0 * (FY0 + FYN)) - 2 / (FY0 * (FY0 + FYS))
            data_2 = 2 / (FY0 * (FY0 + FYN))
            data_3 = 2 / (FY0 * (FY0 + FYS))

            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate(
                (
                    grid.A0.flatten(),
                    grid.AN[grid.A0.flatten()],
                    grid.AS[grid.A0.flatten()],
                )
            )
            data = np.concatenate((data_1, data_2, data_3))

        elif index == 2:
            data_1 = -2 / (FX0 * (FX0 + FXE)) - 2 / (FX0 * (FX0 + FXW))
            data_2 = 2 / (FX0 * (FX0 + FXE))
            data_3 = 2 / (FX0 * (FX0 + FXW))

            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate(
                (
                    grid.A0.flatten(),
                    grid.AE[grid.A0.flatten()],
                    grid.AW[grid.A0.flatten()],
                )
            )
            data = np.concatenate((data_1, data_2, data_3))

        elif index == 3:
            data_1 = -2 / (FZ0 * (FZ0 + FZA)) - 2 / (FZ0 * (FZ0 + FZG))

            mask = grid.AG[grid.A0.flatten()] >= 0

            data_2 = (2 / (FZ0 * (FZ0 + FZG)))[mask]
            data_1[~mask] = (-2 / (FZ0 * (FZ0 + FZA)) - 4 / (FZ0 * (FZ0 + FZG)))[~mask]

            cols_2 = grid.AG[mask]
            rows_2 = grid.A0.flatten()[mask]

            mask = grid.AA[grid.A0.flatten()] <= int(
                grid.N1 * grid.N2 * (grid.N3 - 2) - 1
            )

            data_3 = (2 / (FZ0 * (FZ0 + FZA)))[mask]
            data_1[~mask] = (-4 / (FZ0 * (FZ0 + FZA)) - 2 / (FZ0 * (FZ0 + FZG)))[~mask]

            cols_3 = grid.AA[mask]
            rows_3 = grid.A0.flatten()[mask]

            data = np.concatenate((data_1, data_2, data_3))
            rows = np.concatenate((grid.A0.flatten(), rows_2, rows_3))
            cols = np.concatenate((grid.A0.flatten(), cols_2, cols_3))

        else:
            raise ValueError(f"Invalid index: {index}")

        N = len(grid.A0.flatten())

        M = sps.coo_matrix((data, (rows, cols)), shape=(N, N))
        M.eliminate_zeros()

        return M

    @staticmethod
    def poisson_matrix(grid: Grid) -> sps.dok_matrix:
        """Poisson matrix"""

        FY0 = grid.FY[1:-1].flatten()
        FYN = grid.FY[1:-1, :, grid.north].flatten()
        FYS = grid.FY[1:-1, :, grid.south].flatten()

        FX0 = grid.FX[1:-1].flatten()
        FXE = grid.FX[1:-1, grid.east].flatten()
        FXW = grid.FX[1:-1, grid.west].flatten()

        FZ0 = grid.FZ[1:-1].flatten()
        FZA = grid.FZ[grid.air].flatten()
        FZG = grid.FZ[grid.ground].flatten()

        N = len(grid.A0.flatten())

        # TODO: make this more elegant
        cols_1 = grid.A0.flatten()
        data_1 = (
            -2 / (FY0 * (FY0 + FYN))
            - 2 / (FY0 * (FY0 + FYS))
            - 2 / (FX0 * (FX0 + FXE))
            - 2 / (FX0 * (FX0 + FXW))
            - 2 / (FZ0 * (FZ0 + FZA))
            - 2 / (FZ0 * (FZ0 + FZG))
        )

        cols_2 = grid.AN[grid.A0.flatten()]
        data_2 = 2 / (FY0 * (FY0 + FYN))

        cols_3 = grid.AS[grid.A0.flatten()]
        data_3 = 2 / (FY0 * (FY0 + FYS))

        cols_4 = grid.AE[grid.A0.flatten()]
        data_4 = 2 / (FX0 * (FX0 + FXE))

        cols_5 = grid.AW[grid.A0.flatten()]
        data_5 = 2 / (FX0 * (FX0 + FXW))

        mask = grid.AA[grid.A0.flatten()] <= grid.N1 * grid.N2 * (grid.N3 - 2) - 1

        cols_6 = grid.AA[mask]
        rows_6 = grid.A0.flatten()[mask]
        data_6 = 2 / (FZ0 * (FZ0 + FZA))[mask]

        data_1[~mask] = (
            -2 / (FY0 * (FY0 + FYN))
            - 2 / (FY0 * (FY0 + FYS))
            - 2 / (FX0 * (FX0 + FXE))
            - 2 / (FX0 * (FX0 + FXW))
            - 2 / (FZ0 * (FZ0 + FZG))
        )[~mask]

        mask = grid.AG[grid.A0.flatten()] >= 0

        cols_7 = grid.AG[mask]
        rows_7 = grid.A0.flatten()[mask]
        data_7 = 2 / (FZ0 * (FZ0 + FZG))[mask]

        data_1[~mask] = (
            -2 / (FY0 * (FY0 + FYN))
            - 2 / (FY0 * (FY0 + FYS))
            - 2 / (FX0 * (FX0 + FXE))
            - 2 / (FX0 * (FX0 + FXW))
            - 2 / (FZ0 * (FZ0 + FZA))
        )[~mask]

        i = 0
        j = int(grid.N2 / 2 - 1)
        k = int(round(grid.N1 / 2 - 1))

        data_1[grid.A0[i, j, k]] = (
            -2
            / (
                grid.FY[i + 1, j, k]
                * (grid.FY[i + 1, j, k] + grid.FY[i + 1, j, grid.north[k]])
            )
            - 2
            / (
                grid.FY[i + 1, j, k]
                * (grid.FY[i + 1, j, k] + grid.FY[i + 1, j, grid.south[k]])
            )
            - 2
            / (
                grid.FX[i + 1, j, k]
                * (grid.FX[i + 1, j, k] + grid.FX[i + 1, grid.east[j], k])
            )
            - 2
            / (
                grid.FX[i + 1, j, k]
                * (grid.FX[i + 1, j, k] + grid.FX[i + 1, grid.west[j], k])
            )
            - 2
            / (
                grid.FZ[i + 1, j, k]
                * (grid.FZ[i + 1, j, k] + grid.FZ[grid.air[i], j, k])
            )
            - 4
            / (
                grid.FZ[i + 1, j, k]
                * (grid.FZ[i + 1, j, k] + grid.FZ[grid.ground[i], j, k])
            )
        )

        data = np.concatenate((data_1, data_2, data_3, data_4, data_5, data_6, data_7))
        rows = np.concatenate((np.tile(grid.A0.flatten(), 5), rows_6, rows_7))
        cols = np.concatenate((cols_1, cols_2, cols_3, cols_4, cols_5, cols_6, cols_7))

        M = sps.csr_matrix((data, (rows, cols)), shape=(N, N))

        M.eliminate_zeros()

        return M

    @staticmethod
    def preconditioner(A: sps.spmatrix, N: int) -> spsl.LinearOperator:
        ilu = spsl.spilu(A)
        Mx = lambda x: ilu.solve(x)
        M = spsl.LinearOperator((N, N), Mx)
        return M
