from pydnslab.fields.basefields import Fields
from pydnslab.fields.scipyfields import ScipyFields
from pydnslab.fields.cupy_fields import CupyFields

__all__ = ["get_fields"]


def get_fields(
    dim: tuple, runmode: int, u_nom: float, u_f: float, engine="scipy"
) -> Fields:
    if engine == "scipy":
        return ScipyFields(dim, runmode, u_nom, u_f)
    elif engine == "cupy":
        return CupyFields(dim, runmode, u_nom, u_f)
    else:
        raise NotImplementedError(f"Engine: {engine} not implemented")
