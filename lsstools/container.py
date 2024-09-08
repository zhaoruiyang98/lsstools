from collections import UserDict
from .utils import format_collection
from .static_typing import *


class StaticMapping(Mapping):
    """A static mapping."""

    @classmethod
    def select(cls, keys: Container[str] | None = None, return_dict=False) -> Any:
        all_keys = {
            _: getattr(cls, _)
            for _ in dict.fromkeys(
                _ for _ in dir(cls) if not callable(getattr(cls, _)) and not _.startswith("_")
            )
        }
        if keys is not None:
            all_keys = {k: v for k, v in all_keys.items() if k in keys}
        return all_keys if return_dict else all_keys.values()

    @classmethod
    def keys(cls):
        return cls.select(return_dict=True).keys()

    @classmethod
    def values(cls):
        return cls.select(return_dict=True).values()

    @classmethod
    def items(cls):
        return cls.select(return_dict=True).items()

    def __repr__(self) -> str:
        all_tracers = list(self.select(return_dict=True).keys())
        return f"{type(self).__name__}(" + format_collection(all_tracers) + ")"

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


class NestedDict(UserDict):
    """A dictionary with supports for accessing elements using `x.y.z`-style keys."""

    def __getitem__(self, key) -> Any:
        keys = key.split(".")
        v = self.data
        for i, key in enumerate(keys):
            if not isinstance(v, dict):
                raise TypeError(f"{'.'.join(keys[:i])} is not a dictionary")
            if key in v:
                v = v[key]
                continue
            if hasattr(self.__class__, "__missing__"):
                return self.__class__.__missing__(self, key)  # type: ignore
            raise KeyError(key)
        return v

    def __setitem__(self, key, item):
        keys = key.split(".")
        v = self.data
        entry, entry_name = None, None
        ilast = len(keys) - 1
        for i, key in enumerate(keys):
            if not isinstance(v, dict):
                if entry is not None:
                    del entry[entry_name]
                raise TypeError(f"{'.'.join(keys[:i])} is not a dictionary")
            if i != ilast:
                if key not in v:
                    v[key] = {}
                    if entry is None:
                        entry, entry_name = v, key
                v = v[key]
            else:
                v[key] = item

    def __delitem__(self, key):
        try:
            _ = self[key]
        except (KeyError, TypeError):
            raise
        keys = key.split(".")
        v = self.data
        for key in keys[:-1]:
            v = v[key]
        del v[keys[-1]]
