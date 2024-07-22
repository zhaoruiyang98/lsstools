"""Dirty module for submitting jobs to slurm
"""

import argparse
import inspect
import os
import subprocess
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from ..static_typing import *


_T = TypeVar("_T")


class optional(Generic[_T]):
    def __init__(self, type: Callable[[Any], _T], value=None):
        self.type = type
        self.value = None if value is None else self.type(value)

    def parse(self, value: str):
        if value == "None":
            return type(self)(type=self.type, value=None)
        if self.type == bool:
            if value in ("True", "true"):
                return type(self)(type=self.type, value=True)
            if value in ("False", "false"):
                return type(self)(type=self.type, value=True)
            raise ValueError
        return type(self)(type=self.type, value=self.type(value))


@dataclass
class Scheduler:
    partition: str | None = None
    queue: str | None = None
    nprocs: int = 1
    cpus_per_task: int = 1
    time_limit: str | None = "01:00:00"
    account: str | None = None
    nodes: int = 1
    exclusive: bool = True
    prun: str = "srun"
    prelude: str = ""

    def __post_init__(self):
        if self.partition is None and self.queue is None:
            raise ValueError("should specify partition or queue")
        if self.partition is not None and self.queue is not None:
            raise ValueError("partition and queue are mutually exclusive")

    def header(self):
        s = "#!/bin/bash\n"
        # fmt: off
        s += (
            f"#SBATCH -N {self.nodes}\n"
            f"#SBATCH -n {self.nprocs}\n"
            f"#SBATCH -c {self.cpus_per_task}\n"
            f"#SBATCH --mem=0\n"
        )
        # fmt: on
        if self.partition:
            s += f"#SBATCH -p {self.partition}\n"
        if self.queue:
            s += f"#SBATCH -q {self.queue}\n"
        if self.account:
            s += f"#SBATCH -A {self.account}\n"
        if self.time_limit:
            s += f"#SBATCH -t {self.time_limit}\n"
        if self.exclusive:
            s += f"#SBATCH --exclusive\n"
        return s


class TaskManager:
    def __init__(self, func: Callable, script_path, scheduler: Scheduler, __file__=None):
        self.func = func
        self.script_path = Path(script_path)
        self.scheduler = scheduler
        if __file__ is None:
            __file__ = inspect.stack()[-1].filename
        self.__file__ = __file__
        self.log_dir = self.script_path.parent / f"{self.script_path.stem}"

        all_kwargs = {}
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise ValueError(f"unsupported callable {func}")
            all_kwargs[param.name] = param.default
        self.all_kwargs = all_kwargs
        self.tasks: list[Mapping[str, Any]] = []

    def __call__(self, **kwargs):
        all_kwargs = deepcopy(self.all_kwargs)
        all_kwargs |= kwargs
        if inspect.Parameter.empty in all_kwargs.values():
            raise ValueError(
                f"missing arguments: {[k for k, v in all_kwargs.items() if v is inspect.Parameter.empty]}"
            )
        self.tasks.append(all_kwargs)

    @property
    def first_run(self):
        # XXX: should find a better way to distinguish between the first and second run...
        if self.script_path.exists():
            return False
        else:
            return True

    def run(self, dump_only=False):
        if not self.tasks:
            return
        if self.first_run or dump_only:
            self._dump_script()
            if dump_only:
                return
            else:
                # important to inherit the current environment
                subprocess.run(f"sbatch {self.script_path}", shell=True, env=os.environ)
        else:
            self._run()

    def retry_failed(self):
        new_script = self.log_dir / (self.script_path.stem + "_retry.sh")
        if not new_script.exists():
            failed_idx = self._gather_failed_idx()
            if not failed_idx:
                print("All succeeded!")
                return
            new_python_script = self.log_dir / (Path(self.__file__).stem + "_retry.py")
            shutil.copy(src=self.__file__, dst=new_python_script)
            old_copied_python_script = self.log_dir / Path(self.__file__).name
            with open(self.script_path, "r") as f_in:
                with open(new_script, "w") as f_out:
                    for line in f_in:
                        if not line.startswith("#SBATCH --array"):
                            f_out.write(
                                line.replace(str(old_copied_python_script), str(new_python_script))
                            )
                            continue
                        f_out.write("#SBATCH --array {}\n".format(",".join(map(str, failed_idx))))
            subprocess.run(f"sbatch {new_script}", shell=True, env=os.environ)
        else:
            self._run()

    def _dump_script(self):
        # prepare, mkdir and copy script
        log_dir = self.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        copied_script_file = log_dir / Path(self.__file__).name
        shutil.copy(src=self.__file__, dst=copied_script_file)
        # header
        script = self.scheduler.header()
        output_path = log_dir / f"%a.out.log"
        script += f"#SBATCH --output={output_path}\n"
        error_path = log_dir / f"%a.err.log"
        script += f"#SBATCH --error={error_path}\n"
        if len(self.tasks) == 1:
            script += "#SBATCH --array 0\n"
        else:
            script += f"#SBATCH --array 0-{len(self.tasks) - 1}\n"
        script += "\n"

        # list
        script += "tasks=(\n"
        for task in self.tasks:
            options = '  "'
            for name, value in task.items():
                options += f"--{name} {value} "
            script += f'{options}"\n'
        script += ")\n"

        # launch script
        if self.scheduler.prelude:
            script += self.scheduler.prelude
            if not self.scheduler.prelude.endswith("\n"):
                script += "\n"
        _path = copied_script_file
        if self.scheduler.nodes == self.scheduler.nprocs == self.scheduler.cpus_per_task == 1:
            prog = f"python {_path}"
        else:
            prog = f"{self.scheduler.prun} -n {self.scheduler.nprocs} -c {self.scheduler.cpus_per_task} python -m mpi4py {_path}"
        script += f"{prog} ${{tasks[$SLURM_ARRAY_TASK_ID]}}\n"
        with open(self.script_path, "w") as f:
            f.write(script)

    def _run(self):
        # call self.func, read arguments from command line
        parser = self._get_parser()
        args = parser.parse_args()
        kwargs = {k: getattr(args, k) for k in self.all_kwargs}
        self.func(**kwargs)
        from ..mpi import COMM_WORLD as comm
        from pprint import pprint

        idx = self.tasks.index(kwargs)
        comm.Barrier()
        if comm.rank == 0:
            # token
            with open(self.log_dir / f"success_{idx}.txt", "w") as f:
                pprint(kwargs, stream=f, indent=2, sort_dicts=False)

    def _get_parser(self):
        task = self.tasks[0]
        parser = argparse.ArgumentParser()
        # TODO: better type inference
        for name, value in task.items():
            if value is None:
                raise TypeError(
                    f"`{name}` has default value None, please wrap it with `slurm.optional`"
                )
            parser.add_argument(
                f"--{name}",
                type=value.parse if isinstance(value, optional) else type(value),
                required=True,
            )
        return parser

    def _gather_failed_idx(self):
        ntot = len(self.tasks)
        idx: list[int] = []
        for i in range(ntot):
            if not (self.log_dir / f"success_{i}.txt").exists():
                idx.append(i)
        return idx
