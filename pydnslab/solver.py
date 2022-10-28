import numpy as np

from pydnslab.createfields import Fields
from pydnslab.differentialoperators import Operators
from pydnslab.projection import projection


def solver(
    fields: Fields,
    operators: Operators,
    s: int,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    dt: float,
    nu: float,
    gx: float,
    gy: float,
    gz: float,
):
    fields.u = fields.U.flatten()
    fields.v = fields.V.flatten()
    fields.w = fields.W.flatten()

    uold = fields.u
    vold = fields.v
    wold = fields.w

    uc = fields.u
    vc = fields.v
    wc = fields.w

    uk = np.zeros((fields.N1 * fields.N2 * (fields.N3 - 2), s))
    vk = uk
    wk = uk

    for i in range(s):
        du = np.zeros(fields.N1 * fields.N2 * (fields.N3 - 2))
        dv = du
        dw = du

        if i >= 1:
            for j in range(s):
                du += a[i, j] * uk[:, j]
                dv += a[i, j] * vk[:, j]
                dw += a[i, j] * wk[:, j]
            fields.u = uold + dt * du
            fields.v = vold + dt * dv
            fields.w = wold + dt * dw

            projection(fields, operators)

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

        if i == s:
            fields.u = uc
            fields.v = vc
            fields.w = wc

    projection(fields, operators)

    fields.U = np.reshape(fields.u, np.shape(fields.U))
    fields.V = np.reshape(fields.v, np.shape(fields.V))
    fields.W = np.reshape(fields.w, np.shape(fields.W))

    cox = fields.U * dt / fields.FX[1 : fields.N3 - 1]
    coy = fields.V * dt / fields.FY[1 : fields.N3 - 1]
    coz = fields.W * dt / fields.FZ[1 : fields.N3 - 1]

    co = cox + coy + coz
    comax = np.amax(co)
