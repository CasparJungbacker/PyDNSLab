from petsc4py import PETSc
import numpy as np

from pydnslab.createfields import Fields
from pydnslab.base_operators import Operators


class PetscOperators(Operators):
    def __init__(self, fields: Fields):
        self.Dx = self.differentiate_1(
            fields.N1,
            fields.N2,
            fields.N3,
            fields.FX,
            fields.FY,
            fields.FZ,
            fields.inz,
            fields.inx,
            fields.iny,
            fields.A0,
            fields.AN,
            fields.AS,
            fields.AE,
            fields.AW,
            fields.AA,
            fields.AG,
            fields.east,
            fields.west,
            fields.north,
            fields.south,
            fields.air,
            fields.ground,
            2,
        )

        self.Dy = self.differentiate_1(
            fields.N1,
            fields.N2,
            fields.N3,
            fields.FX,
            fields.FY,
            fields.FZ,
            fields.inz,
            fields.inx,
            fields.iny,
            fields.A0,
            fields.AN,
            fields.AS,
            fields.AE,
            fields.AW,
            fields.AA,
            fields.AG,
            fields.east,
            fields.west,
            fields.north,
            fields.south,
            fields.air,
            fields.ground,
            1,
        )

        self.Dz = self.differentiate_1(
            fields.N1,
            fields.N2,
            fields.N3,
            fields.FX,
            fields.FY,
            fields.FZ,
            fields.inz,
            fields.inx,
            fields.iny,
            fields.A0,
            fields.AN,
            fields.AS,
            fields.AE,
            fields.AW,
            fields.AA,
            fields.AG,
            fields.east,
            fields.west,
            fields.north,
            fields.south,
            fields.air,
            fields.ground,
            3,
        )

    @staticmethod
    def differentiate_1(
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
    ):
        n = N1 * N2 * (N3 - 2)
        M = PETSc.Mat()

        nnz = 3 * np.ones(n, dtype=np.int32)

        M.createAIJ([n, n], nnz=nnz)

        for i in inz - 1:
            for j in inx:
                for k in iny:
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
                        M.setValue(
                            A0[i, j, k],
                            A0[i, j, k],
                            (1 / FY0) * (FYN / (FY0 + FYN) - FYS / (FYS + FY0)),
                        )

                    if index == 2:
                        M.setValue(
                            A0[i, j, k],
                            A0[i, j, k],
                            (1 / FX0) * (FXE / (FX0 + FXE) - FXW / (FXW + FX0)),
                        )
                    if index == 3:
                        M.setValue(
                            A0[i, j, k],
                            A0[i, j, k],
                            (1 / FZ0) * (FZA / (FZ0 + FZA) - FZG / (FZG + FZ0)),
                        )

                    if index == 1:
                        M.setValue(
                            A0[i, j, k], AN[A0[i, j, k]], 1 / FY0 * FY0 / (FY0 + FYN)
                        )
                        M.setValue(
                            A0[i, j, k], AS[A0[i, j, k]], -1 / FY0 * FY0 / (FY0 + FYS)
                        )
                    if index == 2:
                        M.setValue(
                            A0[i, j, k], AE[A0[i, j, k]], 1 / FX0 * FX0 / (FX0 + FXE)
                        )
                        M.setValue(
                            A0[i, j, k], AW[A0[i, j, k]], -1 / FX0 * FX0 / (FX0 + FXW)
                        )
                    if index == 3:
                        if AG[A0[i, j, k]] >= 0:
                            M.setValue(
                                A0[i, j, k],
                                AG[A0[i, j, k]],
                                -1 / FZ0 * FZ0 / (FZ0 + FZG),
                            )

                        else:
                            M.setValue(
                                A0[i, j, k],
                                A0[i, j, k],
                                1
                                / FZ0
                                * (
                                    FZA / (FZ0 + FZA)
                                    - FZG / (FZG + FZ0)
                                    + FZ0 / (FZG + FZ0)
                                ),
                            )
                        if AA[A0[i, j, k]] <= N1 * N2 * (N3 - 2) - 1:
                            M.setValue(
                                A0[i, j, k],
                                AA[A0[i, j, k]],
                                1 / FZ0 * FZ0 / (FZ0 + FZA),
                            )

                        else:
                            M.setValue(
                                A0[i, j, k],
                                A0[i, j, k],
                                1
                                / FZ0
                                * (
                                    FZA / (FZ0 + FZA)
                                    - FZG / (FZG + FZ0)
                                    - FZ0 / (FZA + FZ0)
                                ),
                            )
        M.setUp()
        M.assemble()

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
    ):
        pass

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
    ):
        pass

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
        index: int,
    ):
        pass
