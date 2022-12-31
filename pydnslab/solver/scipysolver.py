import numpy as np
import scipy.sparse.linalg as spsl

import pydnslab.config as config

from pydnslab.solver.basesolver import Solver
from pydnslab.fields.basefields import Fields
from pydnslab.grid import Grid
from pydnslab.operators.scipy_operators import ScipyOperators  # NOT the base class

__all__ = ["ScipySolver"]


class ScipySolver(Solver):
    @staticmethod
    def projection(fields: Fields, operators: ScipyOperators) -> tuple[np.ndarray, ...]:
        div = (
            operators.Dx.dot(fields.u)
            + operators.Dy.dot(fields.v)
            + operators.Dz.dot(fields.w)
        )

        # TODO: throw error on non-zero error code
        p, exit_code = spsl.cg(
            -operators.M, div, x0=fields.pold, tol=1e-3, M=operators.M_inv_approx
        )

        if exit_code != 0:
            raise AssertionError(f"Bicgstab exited with exit code {exit_code}")

        pnew = p

        du = operators.Dxp.dot(p)
        dv = operators.Dyp.dot(p)
        dz = operators.Dzp.dot(p)

        return pnew, du, dv, dz

    @staticmethod
    def adjust_timestep(fields: Fields, grid: Grid, dt: float) -> float:
        cox = fields.U * dt / grid.FX[1 : grid.N3 - 1]
        coy = fields.V * dt / grid.FY[1 : grid.N3 - 1]
        coz = fields.W * dt / grid.FZ[1 : grid.N3 - 1]

        co = cox + coy + coz
        comax = np.amax(co)

        dt = dt / (comax / config.co_target)

        return dt

    @staticmethod
    def timestep(
        fields: Fields,
        operators: ScipyOperators,
        grid: Grid,
        s: int,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        dt: float,
        nu: float,
    ) -> tuple[np.ndarray, ...]:

        uold = np.copy(fields.u)
        vold = np.copy(fields.v)
        wold = np.copy(fields.w)

        uc = np.copy(fields.u)
        vc = np.copy(fields.v)
        wc = np.copy(fields.w)

        uk = np.zeros((grid.N1 * grid.N2 * (grid.N3 - 2), s))
        vk = np.zeros_like(uk)
        wk = np.zeros_like(uk)

        for i in range(s):
            du = np.zeros(grid.N1 * grid.N2 * (grid.N3 - 2))
            dv = np.zeros_like(du)
            dw = np.zeros_like(du)

            if i > 0:
                for j in range(s):
                    du += a[i, j] * uk[:, j]
                    dv += a[i, j] * vk[:, j]
                    dw += a[i, j] * wk[:, j]

                fields.update(du * dt, dv * dt, dw * dt)

                pnew, du, dv, dw = ScipySolver.projection(fields, operators)
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

            gx = 2 * (2 * config.re_tau * config.nu) ** 2 / (config.heigth**3)

            uk[:, i] = -conv_x + diff_x + gx
            vk[:, i] = -conv_y + diff_y + config.gy
            wk[:, i] = -conv_z + diff_z + config.gz

            uc += dt * b[i] * uk[:, i]
            vc += dt * b[i] * vk[:, i]
            wc += dt * b[i] * wk[:, i]

        du = uc - uold
        dv = vc - vold
        dw = wc - wold

        return du, dv, dw
