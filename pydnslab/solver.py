import numpy as np

import pydnslab.config as config

if config.backend == "cupy":
    try:
        import cupy as xp
        import cupyx.scipy.sparse.linalg as spsl
    except ImportError:
        import numpy as xp
        import scipy.sparse.linalg as spsl
else:
    import numpy as xp
    import scipy.sparse.linalg as spsl

from pydnslab.fields import Fields
from pydnslab.grid import Grid
from pydnslab.operators import Operators


class Solver:
    def __init__(self):
        self.s = None
        self.a = None
        self.b = None
        self.c = None

        self._butcher_tableau()

    @staticmethod
    def projection(fields: Fields, operators: Operators) -> tuple[xp.ndarray, ...]:
        div = (
            operators.Dx.dot(fields.u)
            + operators.Dy.dot(fields.v)
            + operators.Dz.dot(fields.w)
        )

        p, _ = spsl.cg(
            -operators.M, div, x0=fields.pold, tol=1e-3, M=operators.M_inv_approx
        )

        pnew = p

        du = operators.Dxp.dot(p)
        dv = operators.Dyp.dot(p)
        dw = operators.Dzp.dot(p)

        return pnew, du, dv, dw

    @staticmethod
    def adjust_timestep(fields: Fields, grid: Grid, dt: float) -> float:
        cox = fields.U * dt / grid.FX[1 : grid.N3 - 1]
        coy = fields.V * dt / grid.FY[1 : grid.N3 - 1]
        coz = fields.W * dt / grid.FZ[1 : grid.N3 - 1]

        co = cox + coy + coz
        comax = xp.amax(co)

        dt = dt / (comax / config.co_target)

        return dt

    def timestep(
        self,
        fields: Fields,
        operators: Operators,
        grid: Grid,
        dt: float,
    ) -> tuple[xp.ndarray, ...]:

        nu = config.nu

        uold = xp.copy(fields.u)
        vold = xp.copy(fields.v)
        wold = xp.copy(fields.w)

        uc = xp.copy(fields.u)
        vc = xp.copy(fields.v)
        wc = xp.copy(fields.w)

        uk = xp.zeros((grid.N1 * grid.N2 * (grid.N3 - 2), self.s))
        vk = xp.zeros_like(uk)
        wk = xp.zeros_like(uk)

        for i in range(self.s):
            du = xp.zeros(grid.N1 * grid.N2 * (grid.N3 - 2))
            dv = xp.zeros_like(du)
            dw = xp.zeros_like(du)

            if i > 0:
                for j in range(self.s):
                    du += self.a[i][j] * uk[:, j]
                    dv += self.a[i][j] * vk[:, j]

                    dw += self.a[i][j] * wk[:, j]

                +operators.Dzz.dot(fields.w)
                fields.update(du * dt, dv * dt, dw * dt)

                pnew, du, dv, dw = Solver.projection(fields, operators)
                fields.update(du, dv, dw, pnew)

            # Convection term
            conv_x = 0.5 * (
                operators.Dx.dot(xp.multiply(fields.u, fields.u))
                + operators.Dy.dot(xp.multiply(fields.v, fields.u))
                + operators.Dzp.dot(xp.multiply(fields.w, fields.u))
                + xp.multiply(fields.u, operators.Dx.dot(fields.u))
                + xp.multiply(fields.v, operators.Dy.dot(fields.u))
                + xp.multiply(fields.w, operators.Dz.dot(fields.u))
            )

            conv_y = 0.5 * (
                operators.Dx.dot(xp.multiply(fields.u, fields.v))
                + operators.Dy.dot(xp.multiply(fields.v, fields.v))
                + operators.Dzp.dot(xp.multiply(fields.w, fields.v))
                + xp.multiply(fields.u, operators.Dx.dot(fields.v))
                + xp.multiply(fields.v, operators.Dy.dot(fields.v))
                + xp.multiply(fields.w, operators.Dz.dot(fields.v))
            )

            conv_z = 0.5 * (
                operators.Dx.dot(xp.multiply(fields.u, fields.w))
                + operators.Dy.dot(xp.multiply(fields.v, fields.w))
                + operators.Dzp.dot(xp.multiply(fields.w, fields.w))
                + xp.multiply(fields.u, operators.Dx.dot(fields.w))
                + xp.multiply(fields.v, operators.Dy.dot(fields.w))
                + xp.multiply(fields.w, operators.Dz.dot(fields.w))
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

            gx = 2 * (2 * config.re_tau * nu) ** 2 / (config.heigth**3)

            uk[:, i] = -conv_x + diff_x + gx
            vk[:, i] = -conv_y + diff_y + config.gy
            wk[:, i] = -conv_z + diff_z + config.gz

            uc += dt * self.b[i] * uk[:, i]
            vc += dt * self.b[i] * vk[:, i]
            wc += dt * self.b[i] * wk[:, i]

        du = uc - uold
        dv = vc - vold
        dw = wc - wold

        return du, dv, dw

    def _butcher_tableau(self) -> None:
        if config.tim == 1:
            self.s = 1
            self.a = [[0]]
            self.b = [1]
            self.c = 0

        elif config.tim == 2:
            self.s = 2
            self.a = [[0, 0], [1, 0]]
            self.b = [0.5, 0.5]
            self.c = [0, 1]

        elif config.tim == 3:
            self.s = 3
            self.a = [[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]]
            self.b = [1 / 6, 2 / 3, 1 / 6]
            self.c = [0, 0.5, 1]

        elif config.tim == 4:
            self.s = 4
            self.a = [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]
            self.b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
            self.c = [0, 0.5, 0.5, 1]

        else:
            raise ValueError(f"Invalid time integration order: {config.tim}")
