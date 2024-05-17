from __future__ import annotations
import logging
from .mpi import COMM_WORLD
from .static_typing import *

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


def get_mpi_logger(name: str, mpicomm: Intracomm = COMM_WORLD) -> MPILogger:
    """Return a logger with the specified name or, if name is None, return a logger which is the root logger of the hierarchy."""
    raw_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(MPILogger)
    logger = cast(MPILogger, MPILogger.manager.getLogger(name))
    logger.mpicomm = mpicomm
    logging.setLoggerClass(raw_logger_class)
    return logger


class MPILogger(logging.Logger):
    """Custom logger class that logs messages based on the rank of the MPI process.

    Parameters
    ----------
    name : str
        Name of logger.
    level : int, optional
        Logging level, default `logging.NOTSET`.
    mpicomm : `mpi4py.MPI.Intracomm`, optional
        Communicator, default `mpi4py.MPI.COMM_WORLD`.
    """

    def __init__(self, name: str, level=logging.NOTSET, mpicomm: Intracomm = COMM_WORLD):
        super().__init__(name, level=level)
        self.mpicomm = mpicomm

    def debug(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().debug(*args, **kwargs)

    def info(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().info(*args, **kwargs)

    def warning(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().warning(*args, **kwargs)

    def warn(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().warn(*args, **kwargs)

    def exception(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().exception(*args, **kwargs)

    def critical(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().critical(*args, **kwargs)

    def fatal(self, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().fatal(*args, **kwargs)

    def log(self, level, msg, *args, rank: int | None = 0, **kwargs):
        if rank is None or rank == self.mpicomm.rank:
            super().log(level=level, msg=msg, *args, **kwargs)
