import numpy as np

from pydnslab.createfields import Fields


def adjust_timestep(fields: Fields, dt: float, co_target: float) -> float:
    cox = fields.U * dt / fields.FX[1 : fields.N3 - 1]
    coy = fields.V * dt / fields.FY[1 : fields.N3 - 1]
    coz = fields.W * dt / fields.FZ[1 : fields.N3 - 1]

    co = cox + coy + coz
    comax = np.amax(co)

    dt = dt / (comax / co_target)

    return dt
