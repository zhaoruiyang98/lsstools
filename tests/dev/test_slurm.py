from lsstools.mpi import COMM_WORLD
from lsstools.ex.slurm import Scheduler, TaskManager


def main(x, y, z="X"):
    from random import random

    if random() < 0.1:
        # 10% random failure
        raise ValueError
    print(f"{x} {y} {z}")


if __name__ == "__main__":
    scheduler = Scheduler(
        partition="sciama2.q",
        nprocs=4,
        cpus_per_task=4,
        time_limit="00:01:00",
    )
    tm = TaskManager(func=main, script_path="jobs/test.sh", scheduler=scheduler, __file__=__file__)
    for x, y in zip([1, 2, 3], ["a", "b", "c"]):
        for z in ["X", "Y"]:
            tm(x=x, y=y, z=z)
    tm.run()
    # tm.retry_failed()
