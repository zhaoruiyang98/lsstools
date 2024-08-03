from __future__ import annotations
import pytest
import typer
from lsstools.ex.slurm import CommandLineFunction
from lsstools.static_typing import *


def test_CommandLineFunction():
    def _fn1(x, y="a"):
        pass

    fn = CommandLineFunction(_fn1)
    assert fn.commandline_callstr(x=3) == "3 --y a"

    def _fn2(x: int, y: str = "x"):
        pass

    fn = CommandLineFunction(_fn2)
    with pytest.raises(KeyError):
        fn.commandline_callstr()
    assert fn.commandline_callstr(x=3) == "3 --y x"
    assert fn.commandline_callstr(x=3, y="fda") == "3 --y fda"

    def _fn3(
        p1: int,
        k1: Annotated[int, typer.Option()],
        p2: Annotated[Optional[str], typer.Argument()] = "p2",
        k2: str = "k2",
    ):
        pass

    fn = CommandLineFunction(_fn3)
    assert fn.commandline_callstr(p1=2, k1=3) == "2 p2 --k1 3 --k2 k2"

    def _fn4(
        p1: bool,
        k1: Annotated[bool, typer.Option()],
        p2: Annotated[bool, typer.Argument()] = False,
        k_2: bool = True,
    ):
        pass

    fn = CommandLineFunction(_fn4)
    assert fn.commandline_callstr(p1=True, k1=False) == "True False --no-k1 --k-2"


if __name__ == "__main__":

    def _fn4(
        p1: bool,
        k1: Annotated[bool, typer.Option()],
        p2: Annotated[bool, typer.Argument()] = False,
        k_2: bool = True,
    ):
        print(f"{p1=} {k1=} {p2=} {k_2=}")

    fn = CommandLineFunction(_fn4)
    fn.run()
