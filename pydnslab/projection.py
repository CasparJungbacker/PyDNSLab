from typing import Any
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from typing import Any

from pydnslab.createfields import Fields
from pydnslab.differentialoperators import Operators


def projection(fields: Fields, operators: Operators, preconditioner: Any) -> None:

    div = (
        operators.Dx.dot(fields.u)
        + operators.Dy.dot(fields.v)
        + operators.Dz.dot(fields.w)
    )

    p, exit_code = spsl.bicgstab(
        -operators.M, div, x0=fields.pold, tol=1e-3, M=preconditioner
    )

    fields.pold = p

    px = operators.Dxp.dot(p)
    py = operators.Dyp.dot(p)
    pz = operators.Dzp.dot(p)

    fields.u -= px
    fields.v -= py
    fields.w -= pz
