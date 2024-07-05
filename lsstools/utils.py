import math
from .static_typing import *


def format_collection(c: Collection, nmax: int = 5, sep: str = ", ") -> str:
    """Format a python collection object.

    Parameters
    ----------
    c : `Collection`
    nmax : int, optional
        The number of objects to shown, by default 5.
    sep : str, optional
        The character used to separate values, by default ", ".

    Returns
    -------
    str
    """
    if isinstance(c, str):
        return c
    type = c.__class__
    if type is list:
        sbegin, send = "[", "]"
    elif type is tuple:
        sbegin, send = "(", ")"
    elif type is set or type is dict:
        sbegin, send = "{", "}"
    elif type is frozenset:
        sbegin, send = "frozenset({", "})"
    elif type is np.ndarray:
        sbegin, send = "array([", "])"
    else:
        sbegin, send = f"{type.__name__}(", ")"

    if len(c) <= nmax:
        smid = sep.join(map(str, c))
    else:
        ileft = math.ceil(nmax / 2)
        iright = len(c) - (nmax - ileft)
        items: list[str] = []
        if isinstance(c, Mapping):
            for i, (k, v) in enumerate(c.items()):
                if i < ileft or i >= iright:
                    items.append(f"{k}: {v}")
        else:
            for i, x in enumerate(c):
                if i < ileft or i >= iright:
                    items.append(str(x))
        items.insert(ileft, "...")
        smid = sep.join(items)
    return f"{sbegin}{smid}{send}"
