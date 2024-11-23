import logging
from lsstools.mpi import COMM_WORLD
from lsstools.log import get_mpi_logger


def test_mpiroot():
    mpicomm = COMM_WORLD
    logger = get_mpi_logger("test")
    logger.setLevel(logging.INFO)
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(logging.Formatter("rank-{rank}: {message}", style="{"))
    logger.addHandler(hdlr)
    logger.info("should be called from rank-0")
    mpicomm.barrier()
    logger.info("should be called from rank-2", mpiroot=2)
    mpicomm.barrier()
    logger.info("should be called from all processors", mpiroot=None)
    mpicomm.barrier()


if __name__ == "__main__":
    test_mpiroot()
