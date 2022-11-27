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
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
            2,
        )

        self.Dyy = self.differentiate_2(
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
            1,
        )

        self.Dzz = self.differentiate_2(
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
            3,
        )

        self.Dxp = self.differentiate_1p(
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
            2,
        )

        self.Dyp = self.differentiate_1p(
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
            1,
        )

        self.Dzp = self.differentiate_1p(
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
            3,
        )

        self.M = self.poisson_matrix(
            grid.N1,
            grid.N2,
            grid.N3,
            grid.FX,
            grid.FY,
            grid.FZ,
            grid.inz,
            grid.inx,
            grid.iny,
            grid.A0,
            grid.AN,
            grid.AS,
            grid.AE,
            grid.AW,
            grid.AA,
            grid.AG,
            grid.east,
            grid.west,
            grid.north,
            grid.south,
            grid.air,
            grid.ground,
        )

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
            data_2 = np.zeros_like(data_1)
            data_3 = np.zeros_like(data_1)
            cols_2 = np.zeros_like(data_1)
            cols_3 = np.zeros_like(data_1)

            a = grid.AG[grid.A0.flatten()] >= 0

            data_2[a] = -1 / FZ0 * FZ0 / (FZ0 + FZG)
            data_2[np.logical_not(a)] = (
                1 / FZ0 * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0) - FZ0 / (FZG + FZ0))
            )

            cols_2[a] = grid.AG[a]
            cols_2[np.logical_not(a)] = grid.A0.flatten(np.logical_not(a))

            a = grid.AA[grid.A0.flatten()] <= int(grid.N1 * grid.N2 * (grid.N3 - 2) - 1)

            data_3[a] = 1 / FZ0 * FZ0 / (FZ0 + FZA)
            data_3[np.logical_not(a)] = (
                1 / FZ0 * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0) + FZ0 / (FZA + FZ0))
            )

            cols_3[a] = grid.AA[a]
            cols_3[np.logical_not(a)] = grid.A0.flatten(np.logical_not(a))

            data = np.concatenate((data_1, data_2, data_3))
            rows = np.tile(grid.A0.flatten(), 3)
            cols = np.concatenate((grid.A0.flatten(), cols_2, cols_3))

        else:
            raise ValueError(f"Invalid index: {index}")

        N = len(grid.A0.flatten())

        M = sps.coo_matrix((data, (rows, cols)), shape=(N, N))
        M.eliminate_zeros()

        return M

    @staticmethod
    def differentiate_1p(
        N1: int,
        N2: int,
        N3: int,
        FX: np.ndarray,
        FY: np.ndarray,
        FZ: np.ndarray,
        inz: np.ndarray,
        inx: np.ndarray,
        iny: np.ndarray,
        A0: np.ndarray,
        AN: np.ndarray,
        AS: np.ndarray,
        AE: np.ndarray,
        AW: np.ndarray,
        AA: np.ndarray,
        AG: np.ndarray,
        east: np.ndarray,
        west: np.ndarray,
        north: np.ndarray,
        south: np.ndarray,
        air: np.ndarray,
        ground: np.ndarray,
        index: int,
    ) -> sps.dok_matrix:
        """First order pressure derivates"""
        m = np.zeros(N1 * N2 * (N3 - 2))
        M = sps.dia_matrix(
            ([m, m, m, m, m, m, m], [-N1 * (N3 - 2), -N1, -1, 0, 1, N1, N1 * (N3 - 2)]),
            (N1 * N2 * (N3 - 2), N1 * N2 * (N3 - 2)),
        ).todok()

        for i in iny:
            for j in inx:
                for k in inz - 1:
                    FY0 = FY[i + 1, j, k]
                    FYN = FY[i + 1, j, north[k]]
                    FYS = FY[i + 1, j, south[k]]

                    FX0 = FX[i + 1, j, k]
                    FXE = FX[i + 1, east[j], k]
                    FXW = FX[i + 1, west[j], k]

                    FZ0 = FZ[i + 1, j, k]
                    FZA = FZ[air[i], j, k]
                    FZG = FZ[ground[k], j, k]

                    if index == 1:
                        M[A0[i, j, k], A0[i, j, k]] = (1 / FY0) * (
                            FYN / (FY0 + FYN) - FYS / (FYS + FY0)
                        )
                    if index == 2:
                        M[A0[i, j, k], A0[i, j, k]] = (1 / FX0) * (
                            FXE / (FX0 + FXE) - FXW / (FXW + FX0)
                        )
                    if index == 3:
                        M[A0[i, j, k], A0[i, j, k]] = (1 / FZ0) * (
                            FZA / (FZ0 + FZA) - FZG / (FZG + FZ0)
                        )

                    if index == 1:
                        M[A0[i, j, k], AN[A0[i, j, k]]] = 1 / FY0 * FY0 / (FY0 + FYN)
                        M[A0[i, j, k], AS[A0[i, j, k]]] = -1 / FY0 * FY0 / (FY0 + FYS)
                    if index == 2:
                        M[A0[i, j, k], AE[A0[i, j, k]]] = 1 / FX0 * FX0 / (FX0 + FXE)
                        M[A0[i, j, k], AW[A0[i, j, k]]] = -1 / FX0 * FX0 / (FX0 + FXW)
                    if index == 3:
                        if AG[A0[i, j, k]] >= 0:
                            M[A0[i, j, k], AG[A0[i, j, k]]] = (
                                -1 / FZ0 * FZ0 / (FZ0 + FZG)
                            )

                        else:
                            M[A0[i, j, k], A0[i, j, k]] = (
                                1
                                / FZ0
                                * (
                                    FZA / (FZ0 + FZA)
                                    - FZG / (FZG + FZ0)
                                    - FZ0 / (FZG + FZ0)
                                )
                            )

                        if AA[A0[i, j, k]] <= N1 * N2 * (N3 - 2) - 1:
                            M[A0[i, j, k], AA[A0[i, j, k]]] = (
                                1 / FZ0 * FZ0 / (FZ0 + FZA)
                            )

                        else:
                            M[A0[i, j, k], A0[i, j, k]] = (
                                1
                                / FZ0
                                * (
                                    FZA / (FZ0 + FZA)
                                    - FZG / (FZG + FZ0)
                                    + FZ0 / (FZA + FZ0)
                                )
                            )
        M = M.tocoo()

        return M

    @staticmethod
    def differentiate_2(
        N1: int,
        N2: int,
        N3: int,
        FX: np.ndarray,
        FY: np.ndarray,
        FZ: np.ndarray,
        inz: np.ndarray,
        inx: np.ndarray,
        iny: np.ndarray,
        A0: np.ndarray,
        AN: np.ndarray,
        AS: np.ndarray,
        AE: np.ndarray,
        AW: np.ndarray,
        AA: np.ndarray,
        AG: np.ndarray,
        east: np.ndarray,
        west: np.ndarray,
        north: np.ndarray,
        south: np.ndarray,
        air: np.ndarray,
        ground: np.ndarray,
        index: int,
    ) -> sps.dok_matrix:
        """Second order derivatives"""

        m = np.zeros(N1 * N2 * (N3 - 2))
        M = sps.dia_matrix(
            ([m, m, m, m, m, m, m], [-N1 * (N3 - 2), -N1, -1, 0, 1, N1, N1 * (N3 - 2)]),
            (N1 * N2 * (N3 - 2), N1 * N2 * (N3 - 2)),
        ).todok()

        for i in iny:
            for j in inx:
                for k in inz - 1:

                    if index == 1:
                        M[A0[i, j, k], A0[i, j, k]] = -2 / (
                            FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, north[k]])
                        ) - 2 / (
                            FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, south[k]])
                        )

                    elif index == 2:
                        M[A0[i, j, k], A0[i, j, k]] = -2 / (
                            FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, east[j], k])
                        ) - 2 / (
                            FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, west[j], k])
                        )

                    elif index == 3:
                        M[A0[i, j, k], A0[i, j, k]] = -2 / (
                            FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k])
                        ) - 2 / (
                            FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[ground[i], j, k])
                        )

                    if index == 1:
                        M[A0[i, j, k], AN[A0[i, j, k]]] = 2 / (
                            FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, north[k]])
                        )
                        M[A0[i, j, k], AS[A0[i, j, k]]] = 2 / (
                            FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, south[k]])
                        )

                    elif index == 2:
                        M[A0[i, j, k], AE[A0[i, j, k]]] = 2 / (
                            FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, east[j], k])
                        )
                        M[A0[i, j, k], AS[A0[i, j, k]]] = 2 / (
                            FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, west[j], k])
                        )
                    elif index == 3:
                        if AG[A0[i, j, k]] >= 0:
                            M[A0[i, j, k], AG[A0[i, j, k]]] = 2 / (
                                FZ[i + 1, j, k]
                                * (FZ[i + 1, j, k] + FZ[ground[i], j, k])
                            )
                        else:
                            M[A0[i, j, k], A0[i, j, k]] = -2 / (
                                FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k])
                            ) - 4 / (
                                FZ[i + 1, j, k]
                                * (FZ[i + 1, j, k] + FZ[ground[i], j, k])
                            )
                        if AA[A0[i, j, k]] <= N1 * N2 * (N3 - 2) - 1:
                            M[A0[i, j, k], AA[A0[i, j, k]]] = 2 / (
                                FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k])
                            )
                        else:
                            M[A0[i, j, k], A0[i, j, k]] = -4 / (
                                FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k])
                            ) - 2 / (
                                FZ[i + 1, j, k]
                                * (FZ[i + 1, j, k] + FZ[ground[i], j, k])
                            )

        M = M.tocoo()

        return M

    @staticmethod
    def poisson_matrix(
        N1: int,
        N2: int,
        N3: int,
        FX: np.ndarray,
        FY: np.ndarray,
        FZ: np.ndarray,
        inz: np.ndarray,
        inx: np.ndarray,
        iny: np.ndarray,
        A0: np.ndarray,
        AN: np.ndarray,
        AS: np.ndarray,
        AE: np.ndarray,
        AW: np.ndarray,
        AA: np.ndarray,
        AG: np.ndarray,
        east: np.ndarray,
        west: np.ndarray,
        north: np.ndarray,
        south: np.ndarray,
        air: np.ndarray,
        ground: np.ndarray,
    ) -> sps.dok_matrix:
        """Poisson matrix"""

        m = np.zeros(N1 * N2 * (N3 - 2))
        M = sps.dia_matrix(
            ([m, m, m, m, m, m, m], [-N1 * (N3 - 2), -N1, -1, 0, 1, N1, N1 * (N3 - 2)]),
            (N1 * N2 * (N3 - 2), N1 * N2 * (N3 - 2)),
        ).todok()

        for i in iny:
            for j in inx:
                for k in inz - 1:
                    M[A0[i, j, k], A0[i, j, k]] = (
                        -2
                        / (FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, north[k]]))
                        - 2
                        / (FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, south[k]]))
                        - 2
                        / (FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, east[j], k]))
                        - 2
                        / (FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, west[j], k]))
                        - 2 / (FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k]))
                        - 2
                        / (FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[ground[i], j, k]))
                    )

                    M[A0[i, j, k], AN[A0[i, j, k]]] = 2 / (
                        FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, north[k]])
                    )
                    M[A0[i, j, k], AS[A0[i, j, k]]] = 2 / (
                        FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, south[k]])
                    )
                    M[A0[i, j, k], AE[A0[i, j, k]]] = 2 / (
                        FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, east[j], k])
                    )
                    M[A0[i, j, k], AW[A0[i, j, k]]] = 2 / (
                        FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, west[j], k])
                    )

                    if AA[A0[i, j, k]] <= N1 * N2 * (N3 - 2) - 1:
                        M[A0[i, j, k], AA[A0[i, j, k]]] = 2 / (
                            FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k])
                        )
                    else:
                        M[A0[i, j, k], A0[i, j, k]] = (
                            -2
                            / (
                                FY[i + 1, j, k]
                                * (FY[i + 1, j, k] + FY[i + 1, j, north[k]])
                            )
                            - 2
                            / (
                                FY[i + 1, j, k]
                                * (FY[i + 1, j, k] + FY[i + 1, j, south[k]])
                            )
                            - 2
                            / (
                                FX[i + 1, j, k]
                                * (FX[i + 1, j, k] + FX[i + 1, east[j], k])
                            )
                            - 2
                            / (
                                FX[i + 1, j, k]
                                * (FX[i + 1, j, k] + FX[i + 1, west[j], k])
                            )
                            - 2
                            / (
                                FZ[i + 1, j, k]
                                * (FZ[i + 1, j, k] + FZ[ground[i], j, k])
                            )
                        )

                    if AG[A0[i, j, k]] >= 1:
                        M[A0[i, j, k], AG[A0[i, j, k]]] = 2 / (
                            FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[ground[i], j, k])
                        )
                    else:
                        M[A0[i, j, k], A0[i, j, k]] = (
                            -2
                            / (
                                FY[i + 1, j, k]
                                * (FY[i + 1, j, k] + FY[i + 1, j, north[k]])
                            )
                            - 2
                            / (
                                FY[i + 1, j, k]
                                * (FY[i + 1, j, k] + FY[i + 1, j, south[k]])
                            )
                            - 2
                            / (
                                FX[i + 1, j, k]
                                * (FX[i + 1, j, k] + FX[i + 1, east[j], k])
                            )
                            - 2
                            / (
                                FX[i + 1, j, k]
                                * (FX[i + 1, j, k] + FX[i + 1, west[j], k])
                            )
                            - 2
                            / (FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k]))
                        )
        i = int(round(N1 / 2) - 1)
        j = int(round(N2 / 2) - 1)
        k = 0

        M[A0[i, j, k], A0[i, j, k]] = (
            -2 / (FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, north[k]]))
            - 2 / (FY[i + 1, j, k] * (FY[i + 1, j, k] + FY[i + 1, j, south[k]]))
            - 2 / (FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, east[j], k]))
            - 2 / (FX[i + 1, j, k] * (FX[i + 1, j, k] + FX[i + 1, west[j], k]))
            - 2 / (FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[air[i], j, k]))
            - 4 / (FZ[i + 1, j, k] * (FZ[i + 1, j, k] + FZ[ground[i], j, k]))
        )

        M = M.tocoo()

        return M

    @staticmethod
    def preconditioner(A: sps.spmatrix, N: int) -> spsl.LinearOperator:
        ilu = spsl.spilu(A)
        Mx = lambda x: ilu.solve(x)
        M = spsl.LinearOperator((N, N), Mx)
        return M
