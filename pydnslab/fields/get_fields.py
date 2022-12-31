import pydnslab.config as config

from pydnslab.fields.basefields import Fields
from pydnslab.fields.scipyfields import ScipyFields
from pydnslab.fields.cupy_fields import CupyFields
from pydnslab.grid import Grid

__all__ = ["get_fields"]


def get_fields(grid: Grid) -> Fields:
    if config.backend == "scipy":
        return ScipyFields(grid)
    elif config.backend == "cupy":
        return CupyFields(grid)
    else:
        raise NotImplementedError(f"Engine: {config.backend} not implemented")
