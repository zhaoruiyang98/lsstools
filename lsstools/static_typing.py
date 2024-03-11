"""Pre-imported symbols from :mod:`collections.abc`, :mod:`typing` and :mod:`numpy.typing` module.

Type alias
----------

.. list-table::
    :widths: 50 50
    
    * - :obj:`ArrayLikeInt`
      - `TypeAlias` of ``Sequence[int] | NDArray[np.integer]``
    * - :obj:`ArrayLikeFloat`
      - `TypeAlias` of ``Sequence[float] | NDArray[np.floating]``
    * - :obj:`ArrayLikeComplex`
      - `TypeAlias` of ``Sequence[complex] | NDArray[np.complexfloating]``

Imported symbols
----------------

.. list-table::
    :widths: 50 50
    
    * - :class:`collections.abc.Callable`
      - ABC for classes that provide the __call__ method.
    * - :class:`collections.abc.Collection`
      - ABC for sized iterable container classes.
    * - :class:`collections.abc.Container`
      - ABC for classes that provide the __contains__() method.
    * - :class:`collections.abc.Hashable`
      - ABC for classes that provide the __hash__() method.
    * - :class:`collections.abc.Iterable`
      - ABC for classes that provide the __iter__() method.
    * - :class:`collections.abc.Iterator`
      - ABC for classes that provide the __iter__() and __next__() methods.
    * - :class:`collections.abc.Sized`
      - ABC for classes that provide the __len__() method.
    * - :class:`collections.abc.Sequence`
      - 
    * - :class:`collections.abc.Set`
      -
    * - :class:`collections.abc.Mapping`
      - 
    * - :class:`collections.abc.MutableMapping`
      - ABCs for read-only and mutable mappings.
    * - :class:`collections.abc.MutableSet`
      - ABCs for read-only and mutable sets.
    * - :class:`collections.abc.MutableSequence`
      - 
    * - :func:`typing.cast`
      - Cast a value to a type.
    * - :func:`typing.overload`
      - Decorator for creating overloaded functions and methods.
    * - :func:`typing.final`
      - Decorator to indicate final methods and final classes.
    * - :func:`typing.runtime_checkable`
      - Mark a protocol class as a runtime protocol.
    * - `typing.Any`
      - Special type indicating an unconstrained type.
    * - :obj:`typing.ClassVar`
      - Special type construct to mark class variables.
    * - :obj:`typing.Concatenate`
      - Special form for annotating higher-order functions.
    * - :obj:`typing.Final`
      - Special typing construct to indicate final names to type checkers.
    * - :class:`typing.Generic`
      - Abstract base class for generic types.
    * - :class:`typing.IO`
      - 
    * - :class:`typing.TextIO`
      - 
    * - :class:`typing.BinaryIO`
      - Generic type IO[AnyStr] and its subclasses TextIO(IO[str]) and BinaryIO(IO[bytes]) represent
        the types of I/O streams such as returned by open().
    * - :obj:`typing.Literal`
      - Special typing form to define “literal types”.
    * - :class:`typing.NamedTuple`
      - Typed version of :func:`collections.namedtuple`.
    * - :obj:`typing.NoReturn`
      - Special type indicating that a function never returns.
    * - :class:`typing.ParamSpec`
      - Parameter specification variable. A specialized version of type variables.
    * - :class:`typing.Protocol`
      - Base class for protocol classes.
    * - :class:`typing.SupportsAbs`
      - An ABC with one abstract method __abs__ that is covariant in its return type.
    * - :class:`typing.SupportsBytes`
      - An ABC with one abstract method __bytes__.
    * - :class:`typing.SupportsComplex`
      - An ABC with one abstract method __complex__.
    * - :class:`typing.SupportsFloat`
      - An ABC with one abstract method __float__.
    * - :class:`typing.SupportsIndex`
      - An ABC with one abstract method __index__.
    * - :class:`typing.SupportsInt`
      - An ABC with one abstract method __int__.
    * - :class:`typing.SupportsRound`
      - An ABC with one abstract method __round__ that is covariant in its return type.
    * - :class:`typing.TypedDict`
      - Special construct to add type hints to a dictionary. At runtime it is a plain dict.
    * - :obj:`typing.TypeAlias`
      - Special annotation for explicitly declaring a type alias.
    * - :obj:`typing.TypeGuard`
      - Special typing construct for marking user-defined type guard functions.
    * - :class:`typing.TypeVar`
      - Type variable.
    * - :obj:`typing.TYPE_CHECKING`
      - A special constant that is assumed to be True by 3rd party static type checkers.
        It is `False` at runtime.
    * - :obj:`numpy.typing.ArrayLike`
      - A Union representing objects that can be coerced into an ndarray.
    * - :obj:`numpy.typing.DTypeLike`
      - A Union representing objects that can be coerced into a dtype.
    * - :obj:`numpy.typing.NDArray`
      - A ``np.ndarray[Any, np.dtype[+ScalarType]]`` type alias generic w.r.t. its dtype.type.

API
---
"""

from __future__ import annotations
import numpy as np
from collections.abc import Callable as Callable
from collections.abc import Collection as Collection
from collections.abc import Container as Container
from collections.abc import Hashable as Hashable
from collections.abc import Iterable as Iterable
from collections.abc import Iterator as Iterator
from collections.abc import Sized as Sized
from collections.abc import Sequence as Sequence
from collections.abc import Set as Set
from collections.abc import Mapping as Mapping
from collections.abc import MutableMapping as MutableMapping
from collections.abc import MutableSet as MutableSet
from collections.abc import MutableSequence as MutableSequence
from typing import cast as cast
from typing import overload as overload
from typing import final as final
from typing import runtime_checkable as runtime_checkable
from typing import Any as Any
from typing import ClassVar as ClassVar
from typing import Concatenate as Concatenate
from typing import Final as Final
from typing import Generic as Generic
from typing import IO as IO
from typing import TextIO as TextIO
from typing import BinaryIO as BinaryIO
from typing import Literal as Literal
from typing import NamedTuple as NamedTuple
from typing import NoReturn as NoReturn
from typing import ParamSpec as ParamSpec
from typing import Protocol as Protocol
from typing import SupportsAbs as SupportsAbs
from typing import SupportsBytes as SupportsBytes
from typing import SupportsComplex as SupportsComplex
from typing import SupportsFloat as SupportsFloat
from typing import SupportsIndex as SupportsIndex
from typing import SupportsInt as SupportsInt
from typing import SupportsRound as SupportsRound
from typing import TypedDict as TypedDict
from typing import TypeAlias as TypeAlias
from typing import TypeGuard as TypeGuard
from typing import TypeVar as TypeVar
from typing import TYPE_CHECKING as TYPE_CHECKING
from numpy.typing import ArrayLike as ArrayLike
from numpy.typing import DTypeLike as DTypeLike
from numpy.typing import NDArray as NDArray


ArrayLikeInt: TypeAlias = Sequence[int] | NDArray[np.integer]
ArrayLikeFloat: TypeAlias = Sequence[float] | NDArray[np.floating]
ArrayLikeComplex: TypeAlias = Sequence[complex] | NDArray[np.complexfloating]


def assert_never(arg: NoReturn, /) -> NoReturn:
    """Ask a static type checker to confirm that a line of code is unreachable.

    Parameters
    ----------
    arg : NoReturn
        Argument to be checked unreachable.

    Returns
    -------
    NoReturn

    Raises
    ------
    AssertionError
        Direct call to this function would raise an :exc:`AssertationError`.
    """
    raise AssertionError("Expected code to be unreachable")
