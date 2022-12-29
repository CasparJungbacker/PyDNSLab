import cupy as cp
import cupyx.scipy.sparse.linalg as spsl
import numpy as np

from pydnslab.solver.basesolver import Solver
from pydnslab.fields.cupy_fields import CupyFields
from pydnslab.grid import Grid
from pydnslab.operators.cupy_operators import CupyOperators

__all__ = ["CupySolver"]


class CupySolver(Solver):
    @staticmethod
    def projection(
        fields: CupyFields, operators: CupyOperators
    ) -> tuple[cp.ndarray, ...]:
        div = (
            operators.Dx.dot(fields.u)
            + operators.Dy.dot(fields.v)
            + operators.Dz.dot(fields.w)
        )

        p = spsl.spsolve(-operators.M, div)

        pnew = p

        du = operators.Dxp.dot(p)
        dv = operators.Dyp.dot(p)
        dw = operators.Dzp.dot(p)

        return pnew, du, dv, dw

    @staticmethod
    def adjust_timestep(
        fields: CupyFields, grid: Grid, dt: float, co_target: float
    ) -> float:
        # TODO: Maybe better to prevent transfer from device to host
        cox = fields.U.get() * dt / grid.FX[1 : grid.N3 - 1]
        coy = fields.V.get() * dt / grid.FY[1 : grid.N3 - 1]
        coz = fields.W.get() * dt / grid.FZ[1 : grid.N3 - 1]

        co = cox + coy + coz
        comax = np.amax(co)

        dt = dt / (comax / co_target)

        return dt

    def timestep(
        self,
        fields: CupyFields,
        operators: CupyOperators,
        grid: Grid,
        s: int,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        dt: float,
        nu: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> tuple[cp.ndarray, ...]:

        uold = cp.copy(fields.u)
        vold = cp.copy(fields.v)
        wold = cp.copy(fields.w)

        uc = cp.copy(fields.u)
        vc = cp.copy(fields.v)
        wc = cp.copy(fields.w)

        uk = cp.zeros((grid.N1 * grid.N2 * (grid.N3 - 2), s))
        vk = cp.zeros_like(uk)
        wk = cp.zeros_like(uk)

        for i in range(s):
            du = cp.zeros(grid.N1 * grid.N2 * (grid.N3 - 2))
            dv = cp.zeros_like(du)
            dw = cp.zeros_like(du)

            if i > 0:
                for j in range(s):
                    du += a[i, j] * uk[:, j]
                    dv += a[i, j] * vk[:, j]
                    dw += a[i, j] * wk[:, j]

                fields.update(du * dt, dv * dt, dw * dt)

                pnew, du, dv, dw = self.projection(fields, operators)
                fields.update(du, dv, dw, pnew)

            # Convection term
            conv_x = 0.5 * (
                operators.Dx.dot(np.multiply(fields.u, fields.u))
                + operators.Dy.dot(np.multiply(fields.v, fields.u))
                + operators.Dzp.dot(np.multiply(fields.w, fields.u))
                + np.multiply(fields.u, operators.Dx.dot(fields.u))
                + np.multiply(fields.v, operators.Dy.dot(fields.u))
                + np.multiply(fields.w, operators.Dz.dot(fields.u))
            )

            conv_y = 0.5 * (
                operators.Dx.dot(np.multiply(fields.u, fields.v))
                + operators.Dy.dot(np.multiply(fields.v, fields.v))
                + operators.Dzp.dot(np.multiply(fields.w, fields.v))
                + np.multiply(fields.u, operators.Dx.dot(fields.v))
                + np.multiply(fields.v, operators.Dy.dot(fields.v))
                + np.multiply(fields.w, operators.Dz.dot(fields.v))
            )

            conv_z = 0.5 * (
                operators.Dx.dot(np.multiply(fields.u, fields.w))
                + operators.Dy.dot(np.multiply(fields.v, fields.w))
                + operators.Dzp.dot(np.multiply(fields.w, fields.w))
                + np.multiply(fields.u, operators.Dx.dot(fields.w))
                + np.multiply(fields.v, operators.Dy.dot(fields.w))
                + np.multiply(fields.w, operators.Dz.dot(fields.w))
            )

            # Diffusion term
            diff_x = nu * (
                operators.Dxx.dot(fields.u)
                + operators.Dyy.dot(fields.u)
                + operators.Dzz.dot(fields.u)
            )
            diff_y = nu * (
                operators.Dxx.dot(fields.v)
                + operators.Dyy.dot(fields.v)
                + operators.Dzz.dot(fields.v)
            )
            diff_z = nu * (
                operators.Dxx.dot(fields.w)
                + operators.Dyy.dot(fields.w)
                + operators.Dzz.dot(fields.w)
            )

            uk[:, i] = -conv_x + diff_x + gx
            vk[:, i] = -conv_y + diff_y + gy
            wk[:, i] = -conv_z + diff_z + gz

            uc += dt * b[i] * uk[:, i]
            vc += dt * b[i] * vk[:, i]
            wc += dt * b[i] * wk[:, i]

        du = uc - uold
        dv = vc - vold
        dw = wc - wold

        return du, dv, dw
