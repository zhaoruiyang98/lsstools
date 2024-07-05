"""Mpi tools.

Constant
--------

.. list-table::
    :widths: 50 50
    
    * - `COMM_WORLD`
      - `mpi4py.MPI.COMM_WORLD`
"""

from __future__ import annotations
from .static_typing import *

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm  # type: ignore
else:
    Intracomm = None


try:
    import mpi4py  # type: ignore
except ImportError:
    MPI_ENABLED = False
else:
    MPI_ENABLED = True

if MPI_ENABLED:
    from mpi4py import MPI  # type: ignore

    COMM_WORLD = MPI.COMM_WORLD
else:
    from unittest.mock import Mock

    COMM_WORLD = cast(Intracomm, Mock())
    COMM_WORLD.rank = 0
    COMM_WORLD.size = 1
    COMM_WORLD.Get_rank.return_value = COMM_WORLD.rank
    COMM_WORLD.Get_size.return_value = COMM_WORLD.size
