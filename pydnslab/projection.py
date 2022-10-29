from typing import Any
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import bicgstab

from typing import Any

from pydnslab.createfields import Fields
from pydnslab.scipy_operators import ScipyOperators


def projection(fields: Fields, operators: ScipyOperators) -> int:

    div = (
        operators.Dx.dot(fields.u)
        + operators.Dy.dot(fields.v)
        + operators.Dz.dot(fields.w)
    )

    p, exit_code = bicgstab(
        -operators.M, div, x0=fields.pold, tol=1e-3, M=operators.M_inv_approx
    )

    fields.pold = p

    px = operators.Dxp.dot(p)
    py = operators.Dyp.dot(p)
    pz = operators.Dzp.dot(p)

    fields.u = fields.u - px
    fields.v = fields.v - py
    fields.w = fields.w - pz

    return exit_code
