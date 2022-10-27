import numpy as np

from pydnslab.createfields import Fields
from pydnslab.differentialoperators import Operators


def solver(
    fields: Fields, s: int, a: np.ndarray, b: np.ndarray, c: np.ndarray, dt: float
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
            u = uold + dt * du
            v = vold + dt * dv
            w = wold + dt * dw
