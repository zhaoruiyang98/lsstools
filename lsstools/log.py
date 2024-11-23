from __future__ import annotations
import contextlib
import logging
from .mpi import COMM_WORLD
from .static_typing import *

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm  # type: ignore


def get_mpi_logger(
    name: str | None = None,
    mpicomm: Intracomm = COMM_WORLD,
    extra: Mapping[str, object] | None = None,
) -> MPILogger:
    """Return a logger with the specified name or, if name is None, return a logger which is the root logger of the hierarchy."""
    logger = logging.getLogger(name=name)
    logger.addFilter(MPIFilter(mpicomm=mpicomm))
    return MPILogger(logger=logger, mpicomm=mpicomm, extra=extra)


@contextlib.contextmanager
def log_root_only(logroot: int = 0, level: int = logging.INFO, mpicomm: Intracomm | None = None):
    """Disable logging when mpicomm.rank != logroot."""
    if mpicomm is None:
        mpicomm = COMM_WORLD
    if mpicomm.rank != logroot:
        logging.disable(level)
    try:
        yield
    finally:
        if mpicomm.rank != logroot:
            logging.disable(logging.NOTSET)


class MPIFilter(logging.Filter):
    """Attach attribute ``rank`` to records."""

    def __init__(self, name: str = "", mpicomm: Intracomm | None = None) -> None:
        """
        Initialize a filter.

        Initialize with the name of the logger which, together with its
        children, will have its events allowed through the filter. If no
        name is specified, allow every event.
        """
        super().__init__(name)
        if mpicomm is None:
            mpicomm = COMM_WORLD
        self.mpicomm = mpicomm

    def filter(self, record):
        """
        Determine if the specified record is to be logged.

        Returns True if the record should be logged, or False otherwise.
        If deemed appropriate, the record may be modified in-place.
        """
        record.rank = self.mpicomm.rank
        return super().filter(record)


class MPILogger(logging.LoggerAdapter):
    """Custom logger adapter class that logs messages based on the rank of the MPI process.

    `MIPLogger.log` accepts an optional keyword `mpiroot`, by default `mpiroot` is 0. If `mpiroot`
    is set to `None`, log information from all possible mpi processors.

    Parameters
    ----------
    logger : logging.Logger
        The underlying logger.
    mpicomm : `mpi4py.MPI.Intracomm`, optional
        Communicator, default `mpi4py.MPI.COMM_WORLD`.
    extra : str to object mapping, optional
        Additional `LogRecord` attributes.
    """

    def __init__(
        self,
        logger: logging.Logger,
        mpicomm: Intracomm = COMM_WORLD,
        extra: Mapping[str, object] | None = None,
    ):
        self.logger: logging.Logger
        super().__init__(logger=logger, extra=extra)
        self.mpicomm = mpicomm

    def log(self, level, msg, *args, **kwargs):
        """
        Delegate a log call to the underlying logger, after adding
        contextual information from this adapter instance.
        """
        mpiroot = kwargs.pop("mpiroot", 0)
        if (mpiroot is None) or (self.mpicomm.rank == mpiroot):
            if self.isEnabledFor(level):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)

    # additional methods to mimick the logger class
    def addHandler(self, hdlr):
        """
        Add the specified handler to this logger.
        """
        return self.logger.addHandler(hdlr)

    def removeHandler(self, hdlr):
        """
        Remove the specified handler from this logger.
        """
        return self.logger.removeHandler(hdlr)

    def addFilter(self, filter):
        """
        Add the specified filter to this handler.
        """
        return self.logger.addFilter(filter)

    def removeFilter(self, filter):
        """
        Remove the specified filter from this handler.
        """
        return self.logger.removeFilter(filter)
