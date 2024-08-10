from __future__ import annotations
import numpy as np
from copy import deepcopy
from numpy.lib.mixins import NDArrayOperatorsMixin
from ..static_typing import *
from ..utils import format_collection


def monotonic(array):
    diff = np.diff(array)
    return np.all(diff > 0) or np.all(diff < 0)


class NDArrayAttributesMixin:
    _array: str

    def __init_subclass__(cls, array: str = "data") -> None:
        cls._array = array

    @property
    def dtype(self) -> np.dtype:
        return getattr(self, self._array).dtype

    @property
    def size(self) -> int:
        return getattr(self, self._array).size

    @property
    def itemsize(self) -> int:
        return getattr(self, self._array).itemsize

    @property
    def nbytes(self) -> int:
        return getattr(self, self._array).nbytes

    @property
    def ndim(self) -> int:
        return getattr(self, self._array).ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return getattr(self, self._array).shape

    @property
    def strides(self) -> tuple[int, ...]:
        return getattr(self, self._array).strides


class ArrayEdges(NDArrayOperatorsMixin, NDArrayAttributesMixin, array="value"):
    @overload
    def __init__(self, mid, /): ...
    @overload
    def __init__(self, lower, upper, /): ...
    @overload
    def __init__(self, lower, upper, mid, /, *, value=None): ...
    def __init__(self, *args, value=None):
        if len(args) == 1:
            lower = np.asarray(args[0])
            upper, mid = lower.copy(), lower.copy()
        elif len(args) == 2:
            lower, upper = map(np.asarray, args)
            mid = 0.5 * (lower + upper)
        elif len(args) == 3:
            lower, upper, mid = map(np.asarray, args)
        else:
            raise ValueError
        self.lower = lower
        self.upper = upper
        self.mid = mid
        self.value = np.asarray(value) if value is not None else self.mid.copy()
        if not (self.lower.ndim == self.upper.ndim == self.mid.ndim == 1 == self.value.ndim):
            raise ValueError
        if np.unique(self.mid).shape != self.mid.shape:
            raise ValueError

    @classmethod
    def _direct_init(cls, lower, upper, mid, value):
        obj = cls.__new__(cls)
        obj.lower = lower
        obj.upper = upper
        obj.mid = mid
        obj.value = value
        return obj

    def __iter__(self):
        return iter((self.lower, self.upper, self.mid))

    def __getitem__(self, i):
        return type(self)(self.lower[i], self.upper[i], self.mid[i], value=self.value[i])

    def __repr__(self):
        s = format_collection(self.mid)[1 + len("array") : -1]
        return f"{type(self).__name__}({s})"

    # ndarray api
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            nself = sum(isinstance(_, ArrayEdges) for _ in inputs)
            if nself != 1:
                return NotImplemented
            inputs = []
            for x in inputs:
                if isinstance(x, FramedArray):
                    x = x.value
                if isinstance(x, ArrayEdges):
                    x = x.value
                inputs.append(x)
            return type(self)._direct_init(
                self.lower.copy(),
                upper=self.upper.copy(),
                mid=self.mid.copy(),
                value=ufunc(*inputs, **kwargs),
            )
        else:
            return NotImplemented


class FramedArrayILocator:
    def __init__(self, fa: FramedArray):
        self.fa = fa

    def __getitem__(self, args):
        fa = self.fa
        value = fa.value[args]
        edges: dict[str, ArrayEdges] = {}
        iedges = 0
        edges_names = list(fa.edges.keys())

        def slice_edges(arg):
            name = edges_names[iedges]
            x = fa.edges[name][arg]
            if x.mid.size != 0:
                edges[name] = x

        args = args if isinstance(args, tuple) else (args,)
        if ... in args:
            # already valid
            if len(args) == 1:
                args = (slice(None, None, None),)
            else:
                ie = args.index(...)
                tmp = (slice(None, None, None),) * (len(fa.edges) - len(args) + 1)
                args = args[:ie] + tmp + args[ie + 1 :]
        for arg in args:
            if isinstance(arg, FramedArray):
                raise NotImplementedError
            elif isinstance(arg, (np.ndarray, np.generic)):
                if arg.shape == ():
                    if not np.issubdtype(arg.dtype, np.integer):
                        raise TypeError("expected integer index")
                    # skip
                elif arg.ndim == 1 and arg.size == 1:
                    if not np.issubdtype(arg.dtype, np.integer):
                        raise TypeError("expected integer index")
                    slice_edges(arg)
                else:
                    raise NotImplementedError
            elif isinstance(arg, list):
                if len(arg) == 1:
                    if not isinstance(arg[0], int):
                        raise TypeError("expected integer index")
                    slice_edges(arg)
                else:
                    raise NotImplementedError
            elif isinstance(arg, slice):
                slice_edges(arg)
            elif isinstance(arg, int):
                pass  # skip
            else:
                raise NotImplementedError
            iedges += 1
        edges |= {name: fa.edges[name] for name in edges_names[iedges:]}
        assert value.shape == tuple([v.mid.size for v in edges.values()])
        return type(fa)(value, edges)


class FramedArrayLocator:
    def __init__(self, fa: FramedArray):
        self.fa = fa

    def search_index(self, edges: ArrayEdges, value, method="nearest"):
        if edges.mid.dtype.char in np.typecodes["AllFloat"]:
            if method == "nearest":
                idx = np.argmin(np.abs(edges.mid - value))
            elif method == "left":
                if not monotonic(edges.mid):
                    raise IndexError
                idx = np.searchsorted(edges.mid, value, side="left")
            elif method == "right":
                if not monotonic(edges.mid):
                    raise IndexError
                idx = np.searchsorted(edges.mid, value, side="right")
            else:
                raise ValueError
        else:
            # direct search
            idx = -1
            for i, x in enumerate(edges.mid):
                if x == value:
                    idx = i
                    break
            if idx < 0:
                raise IndexError
            if method == "right":
                idx += 1
        return idx

    def __getitem__(self, args):
        fa = self.fa
        iedges = 0
        edges_names = list(fa.edges.keys())
        args = args if isinstance(args, tuple) else (args,)
        if (ne := args.count(...)) == 1:
            if len(args) == 1:
                args = (slice(None, None, None),)
            else:
                ie = args.index(...)
                tmp = (slice(None, None, None),) * (len(fa.edges) - len(args) + 1)
                args = args[:ie] + tmp + args[ie + 1 :]
        elif ne > 1:
            raise IndexError
        iargs = []
        for arg in args:
            aedge = fa.edges[edges_names[iedges]]
            dtype = aedge.mid.dtype
            if isinstance(arg, list):
                if len(arg) != 1:
                    raise NotImplementedError
                iargs.append([self.search_index(aedge, arg[0], method="nearest")])
            elif isinstance(arg, slice):
                if arg.step is not None:
                    raise NotImplementedError
                if arg == slice(None, None, None):
                    iargs.append(arg)
                    continue
                istart, istop = None, None
                if arg.start is not None:
                    istart = self.search_index(aedge, arg.start, method="left")
                if arg.stop is not None:
                    istop = self.search_index(aedge, arg.stop, method="right")
                iargs.append(slice(istart, istop))
            elif (v := np.array(arg, dtype=dtype)).shape == ():
                iargs.append(self.search_index(aedge, v, method="nearest"))
            else:
                raise NotImplementedError
            iedges += 1
        return self.fa[tuple(iargs)]


class FramedArray(NDArrayOperatorsMixin, NDArrayAttributesMixin, array="value"):
    """Numpy-like array with edges.

    See Also
    --------
    API design is motivated by xarray.
    """

    def __init__(
        self, value: ArrayLike, edges: Mapping[str, ArrayEdges | tuple[ArrayLike] | ArrayLike]
    ):
        self.value: NDArrayComplex = np.asarray(value)
        self.edges: Mapping[str, ArrayEdges] = {}
        for k, v in edges.items():
            if isinstance(v, ArrayEdges):
                self.edges[k] = v
            elif isinstance(v, tuple):
                if len(v) == 4:
                    self.edges[k] = ArrayEdges(*v[:-1], value=v[-1])
                else:
                    self.edges[k] = ArrayEdges(*v)
            else:
                self.edges[k] = ArrayEdges(v)
        self.dims: tuple[str, ...] = tuple(self.edges.keys())
        self.coords: Mapping[str, NDArray] = {k: v.mid for k, v in self.edges.items()}
        if self.value.shape != tuple([v.mid.size for v in self.edges.values()]):
            raise ValueError(f"edges does not match value")

    @property
    def loc(self):
        """positional label-based index"""
        return FramedArrayLocator(self)

    @property
    def iloc(self):
        """positional integer-based index"""
        return FramedArrayILocator(self)

    def sel(self, **kwargs):
        """label-based index by name"""
        return self.loc[self._sel(**kwargs)]

    def isel(self, **kwargs):
        """integer-based index by name"""
        return self.iloc[self._sel(**kwargs)]

    def __getitem__(self, args):
        """positional integer-based index"""
        return self.iloc[args]

    def __repr__(self):
        s = ", ".join([f"{k}: {v.size}" for k, v in self.edges.items()])
        return f"<{type(self).__name__} ({s})>"

    def vec(self, ignore_nan=True):
        if ignore_nan:
            return np.ravel(self.value[np.isnan(self.value)])
        return np.ravel(self.value)

    def where(self, cond, drop=False):
        raise NotImplementedError

    def pipe(self, func, *args, **kwargs):
        return type(self)(func(self.value, *args, **kwargs), self.edges)

    def _sel(self, **kwargs):
        edges_names = list(self.edges.keys())
        idx_mapping = {edges_names.index(k): v for k, v in kwargs.items()}
        imax = max(idx_mapping.keys()) + 1
        args = []
        for i in range(imax):
            args.append(idx_mapping.get(i, slice(None, None, None)))
        return tuple(args)

    # ndarray api
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            nself = sum(isinstance(_, FramedArray) for _ in inputs)
            if nself != 1:
                return NotImplemented
            inputs = []
            for x in inputs:
                if isinstance(x, FramedArray):
                    x = x.value
                if isinstance(x, ArrayEdges):
                    x = x.value
                inputs.append(x)
            return type(self)(ufunc(*inputs, **kwargs), deepcopy(self.edges))
        else:
            return NotImplemented

    def __iter__(self):
        return iter(self.value)
