from __future__ import annotations
import itertools
import numpy as np
from copy import deepcopy
from numpy.lib.mixins import NDArrayOperatorsMixin
from ..static_typing import *
from ..utils import format_collection, Picklable


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


class ArrayEdges(Picklable, NDArrayOperatorsMixin, NDArrayAttributesMixin, array="value"):
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

    def __iter__(self):
        return iter((self.lower, self.upper, self.mid))

    def __getitem__(self, i):
        return type(self)(self.lower[i], self.upper[i], self.mid[i], value=self.value[i])

    def __repr__(self):
        s = format_collection(self.mid)[1 + len("array") : -1]
        return f"{type(self).__name__}({s})"

    @classmethod
    def _direct_init(cls, lower, upper, mid, value):
        obj = cls.__new__(cls)
        obj.lower = lower
        obj.upper = upper
        obj.mid = mid
        obj.value = value
        return obj

    @classmethod
    def average(cls, arrays: list[ArrayEdges]):
        if arrays[0].value.dtype.char in (np.typecodes["AllFloat"] + np.typecodes["AllInteger"]):
            value = np.mean([_.value for _ in arrays], axis=0)
        else:
            value = arrays[0].value
        return cls._direct_init(
            lower=arrays[0].lower, upper=arrays[0].upper, mid=arrays[0].mid, value=value
        )

    # ndarray api
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            nself = sum(isinstance(_, ArrayEdges) for _ in inputs)
            if nself != 1:
                return NotImplemented
            inputs2 = []
            for x in inputs:
                if isinstance(x, FramedArray):
                    x = x.value
                if isinstance(x, ArrayEdges):
                    x = x.value
                inputs2.append(x)
            return type(self)._direct_init(
                self.lower.copy(),
                upper=self.upper.copy(),
                mid=self.mid.copy(),
                value=ufunc(*inputs2, **kwargs),
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
                    iedges += 1
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


class FramedArray(Picklable, NDArrayOperatorsMixin, NDArrayAttributesMixin, array="value"):
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
            return np.ravel(self.value[~np.isnan(self.value)])
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

    @classmethod
    def average(cls, arrays: list[FramedArray]):
        edge_names = arrays[0].edges.keys()
        edges = {name: ArrayEdges.average([_.edges[name] for _ in arrays]) for name in edge_names}
        value = np.mean([_.value for _ in arrays], axis=0)
        return cls(value, edges)

    # ndarray api
    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            nself = sum(isinstance(_, FramedArray) for _ in inputs)
            if nself != 1:
                return NotImplemented
            inputs2 = []
            for x in inputs:
                if isinstance(x, FramedArray):
                    x = x.value
                if isinstance(x, ArrayEdges):
                    x = x.value
                inputs2.append(x)
            return type(self)(ufunc(*inputs2, **kwargs), deepcopy(self.edges))
        else:
            return NotImplemented

    def __iter__(self):
        return iter(self.value)


def _paint_text(
    ax, text, loc="lower right", x=None, y=None, va=None, ha=None, fontsize="medium", **kwargs
):
    match loc:
        case "lower right":
            x = x or 0.95
            y = y or 0.05
            va = va or "bottom"
            ha = ha or "right"
        case "top right":
            x = x or 0.95
            y = y or 0.95
            va = va or "top"
            ha = ha or "right"
        case "lower left":
            x = x or 0.05
            y = y or 0.05
            va = va or "bottom"
            ha = ha or "left"
        case "top left":
            x = x or 0.05
            y = y or 0.95
            va = va or "top"
            ha = ha or "left"
        case _:
            raise NotImplementedError
    ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=fontsize, **kwargs)
    return ax


class CovarianceMatrix(Picklable, NDArrayAttributesMixin, array="value"):
    def __init__(self, fa: FramedArray, axes1: tuple[str, ...], axes2: tuple[str, ...]):
        # XXX: the limitation is it only applies to the multi-tracer case, but not P+B because of different indices
        self.fa = fa
        self.axes1 = axes1
        self.axes2 = axes2
        if common := set(self.axes1).intersection(self.axes2):
            s = "axes" if len(common) > 1 else "axis"
            raise ValueError(f"it is not allowed to have common {s} {common}")
        if edges := set(self.axes1 + self.axes2).difference(self.fa.edges.keys()):
            raise ValueError(f"missing edges: {edges}")
        for x1, x2 in zip(self.axes1, self.axes2):
            if self.fa.edges[x1].size != self.fa.edges[x2].size:
                raise ValueError(f"size of edge {x1} and {x2} does not match")
        edges_names = list(self.fa.edges.keys())
        src = [edges_names.index(_) for _ in itertools.chain(self.axes1, self.axes2)]
        dest = [*range(len(self.axes1) + len(self.axes2))]
        value = np.moveaxis(self.fa.value, src, dest)
        edges = {_: self.fa.edges[_] for _ in itertools.chain(self.axes1, self.axes2)}
        self.fa = FramedArray(value, edges)
        nlen = np.prod([edges[_].size for _ in self.axes1])
        self.value = self.fa.value.reshape(nlen, -1)

    def select(self, **kwargs):
        axes = self.axes1 + self.axes2
        kwargs = {_: kwargs.get(_, slice(None)) for _ in axes}
        fa = self.fa.sel(**kwargs)
        axes1 = tuple([_ for _ in self.axes1 if _ in fa.edges])
        axes2 = tuple([_ for _ in self.axes2 if _ in fa.edges])
        return type(self)(fa, axes1, axes2)

    def to_corr(self):
        cov = self.value
        d = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(d, d)
        corr[~np.isfinite(corr)] = 0.0
        corr_array = corr.reshape(self.fa.shape)
        fa = FramedArray(corr_array, edges=self.fa.edges)
        return type(self)(fa, self.axes1, self.axes2)

    def corrplot(self, cov2: CovarianceMatrix | None = None, labels=None, show=True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        corr1 = self.to_corr().value
        corr2 = corr1 if cov2 is None else cov2.to_corr().value
        corr = np.triu(corr1) + np.tril(corr2, k=-1)
        fig, ax = plt.subplots()
        if labels:
            if isinstance(labels, (tuple, list)):
                _paint_text(ax, labels[0], loc="lower right", fontsize="x-large")
                if cov2 is not None:
                    _paint_text(ax, labels[1], loc="top left", fontsize="x-large")
            else:
                _paint_text(ax, labels, loc="lower right", fontsize="x-large")
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="bwr", origin="lower")
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im, cax=cax)
        if show:
            plt.show()
        return fig, ax

    def __repr__(self) -> str:
        edges = self.fa.edges
        axes1 = self.axes1
        axes2 = self.axes2
        s1 = ", ".join([f"{k}: {v.size}" for k, v in zip(axes1, [edges[_] for _ in axes1])])
        s2 = ", ".join([f"{k}: {v.size}" for k, v in zip(axes2, [edges[_] for _ in axes2])])
        return f"<{type(self).__name__} ({s1}) ({s2})>"

    @classmethod
    def from_arrays(cls, arrays: list[FramedArray], axes1: tuple[str, ...], axes2: tuple[str, ...]):
        data_vector = [x.vec(ignore_nan=False) for x in arrays]
        cov = np.cov(data_vector, rowvar=False)
        shape = arrays[0].shape
        edge_names = arrays[0].edges.keys()
        edge_values = [ArrayEdges.average([_.edges[name] for _ in arrays]) for name in edge_names]
        edges = {
            k: v
            for k, v in zip(
                itertools.chain(axes1, axes2), itertools.chain(edge_values, edge_values)
            )
        }
        value = FramedArray(cov.reshape(shape + shape), edges=edges)
        return cls(value, axes1, axes2)
