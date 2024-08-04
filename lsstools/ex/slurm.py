"""Dirty module for submitting jobs to slurm
"""

import inspect
import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints
from ..static_typing import *


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
    driver: str = "srun"
    bash_path: str = "/bin/bash"
    prelude: str = ""

    def __post_init__(self):
        if self.partition is None and self.queue is None:
            raise ValueError("should specify partition or queue")
        if self.partition is not None and self.queue is not None:
            raise ValueError("partition and queue are mutually exclusive")

    def header(self):
        s = f"#!{self.bash_path}\n"
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


def is_typer_argument(hint):
    import typer

    if metadata := getattr(hint, "__metadata__", None):
        return isinstance(metadata[0], type(typer.Argument()))
    return False


def is_typer_option(hint):
    import typer

    if metadata := getattr(hint, "__metadata__", None):
        return isinstance(metadata[0], type(typer.Option()))
    return False


class CommandLineFunction:
    """Function supported by slurm.py

    Attributes
    ----------
    func : Callable
        main function
    arg_names : list[str]
        CLI positional argument names
    option_names : list[str]
        CLI optional argument names
    defaults : dict[str, Any]
        default values
    """

    def __init__(self, func) -> None:
        self.func = func
        self.arg_names: list[str] = []
        self.option_names: list[str] = []
        self.defaults: dict[str, Any] = {}

        # inspect func
        sig = inspect.signature(func)
        hints = get_type_hints(self.func, include_extras=True)
        for param in sig.parameters.values():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise ValueError(f"unsupported callable {self.func}")
            if (hint := hints.get(param.name)) and hasattr(hint, "__metadata__"):
                # annotated hint might be typer.Argument or typer.Option
                if is_typer_argument(hint):
                    self.arg_names.append(param.name)
                elif is_typer_option(hint):
                    self.option_names.append(param.name)
                else:
                    assert_never(hint)
            else:
                if param.kind is param.POSITIONAL_OR_KEYWORD:
                    if param.default is inspect.Parameter.empty:
                        self.arg_names.append(param.name)
                    else:
                        self.option_names.append(param.name)
                if param.kind is param.KEYWORD_ONLY:
                    self.option_names.append(param.name)
            if param.default is not inspect.Parameter.empty:
                self.defaults[param.name] = param.default

    def commandline_callstr(self, **kwargs):
        s = []
        for param in self.arg_names:
            v = kwargs[param] if param in kwargs else self.defaults[param]
            s += [str(v)]
        for param in self.option_names:
            v = kwargs[param] if param in kwargs else self.defaults[param]
            param = param.replace("_", "-")
            if v is True:
                s += [f"--{param}"]
            elif v is False:
                s += [f"--no-{param}"]
            else:
                s += [f"--{param}", str(v)]
        return " ".join(s)

    def run(self):
        import typer

        typer.run(self.func)


_P = ParamSpec("_P")
_T = TypeVar("_T")


class TaskManager(Generic[_P, _T]):
    def __init__(self, func: Callable[_P, _T], script_path, scheduler: Scheduler, pyfile=None):
        self.func = CommandLineFunction(func)
        self.script_path = Path(script_path)
        self.scheduler = scheduler
        self.pyfile = inspect.stack()[-1].filename if pyfile is None else pyfile
        self.log_dir = self.script_path.parent / f"{self.script_path.stem}"

        self.tasks: list[Mapping[str, Any]] = []

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs):
        if args:
            raise TypeError(f"positional arguments are not allowed")
        # check if there are any missing arguments
        all_parameters = set(self.func.arg_names + self.func.option_names)
        if missing := all_parameters.difference(self.func.defaults | kwargs):
            raise ValueError(f"missing arguments: {missing}")
        self.tasks.append(kwargs)

    def run(self, dump_only: bool = False, direct: int | bool = False):
        if not self.tasks:
            return
        if self._is_first_run or dump_only:
            self._dump_script()
            if dump_only:
                return
            # important to inherit the current environment
            options: Any = dict(shell=True, env=os.environ)
            if direct is False:
                subprocess.run(f"sbatch {self.script_path}", **options)
            elif direct is True:
                for i in range(len(self.tasks)):
                    subprocess.run(
                        f"export SLURM_ARRAY_TASK_ID={i}; bash {self.script_path}", **options
                    )
            else:
                subprocess.run(
                    f"export SLURM_ARRAY_TASK_ID={direct}; bash {self.script_path}", **options
                )
        else:
            self.func.run()

    def retry_failed(self, direct: int | bool = False):
        new_script = self.log_dir / (self.script_path.stem + "_retry.sh")
        if not new_script.exists():
            failed_idx = self._gather_failed_idx()
            if not failed_idx:
                print("All succeeded!")
                return
            new_python_script = self.log_dir / (Path(self.pyfile).stem + "_retry.py")
            shutil.copy(src=self.pyfile, dst=new_python_script)
            old_copied_python_script = self.log_dir / Path(self.pyfile).name
            with open(self.script_path, "r") as f_in:
                with open(new_script, "w") as f_out:
                    for line in f_in:
                        if not line.startswith("#SBATCH --array"):
                            f_out.write(
                                line.replace(str(old_copied_python_script), str(new_python_script))
                            )
                            continue
                        f_out.write("#SBATCH --array {}\n".format(",".join(map(str, failed_idx))))
            options: Any = dict(shell=True, env=os.environ)
            if direct is False:
                subprocess.run(f"sbatch {new_script}", **options)
            elif direct is True:
                for i in failed_idx:
                    subprocess.run(f"export SLURM_ARRAY_TASK_ID={i}; bash {new_script}", **options)
            else:
                subprocess.run(f"export SLURM_ARRAY_TASK_ID={direct}; bash {new_script}", **options)
        else:
            self.func.run()

    @property
    def _is_first_run(self):
        # XXX: should find a better way to distinguish between the first and second run...
        if self.script_path.exists():
            return False
        else:
            return True

    def _dump_script(self):
        # prepare, mkdir and copy script
        log_dir = self.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        copied_script_file = log_dir / Path(self.pyfile).name
        shutil.copy(src=self.pyfile, dst=copied_script_file)
        # extra header
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
            cmd = self.func.commandline_callstr(**task)
            script += f'  "{cmd}"\n'
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
            prog = f"{self.scheduler.driver} -n {self.scheduler.nprocs} -c {self.scheduler.cpus_per_task} python -m mpi4py {_path}"
        script += f'PROG="{prog}"\n'
        script += "COMMAND=${tasks[$SLURM_ARRAY_TASK_ID]}\n"
        script += f"SUCCESS_TOKEN={str(self.log_dir / 'success_')}${{SLURM_ARRAY_TASK_ID}}.txt\n"
        script += '${PROG} ${COMMAND} && echo "${COMMAND}" > ${SUCCESS_TOKEN}\n'
        with open(self.script_path, "w") as f:
            f.write(script)

    def _gather_failed_idx(self):
        ntot = len(self.tasks)
        idx: list[int] = []
        for i in range(ntot):
            if not (self.log_dir / f"success_{i}.txt").exists():
                idx.append(i)
        return idx
