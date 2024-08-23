from __future__ import annotations
import numpy as np
import pytest
from lsstools.container import NestedDict
from lsstools.ex.container import ArrayEdges, FramedArray


class XFailContext:
    def __enter__(self):
        pass

    def __exit__(self, type, val, traceback):
        if type is not None:
            pytest.xfail(str(val))


xfail = XFailContext()


class TestNestedDict:
    def test_set_and_get_item(self):
        mapping = NestedDict()
        mapping["key1"] = "value1"
        assert mapping["key1"] == "value1"
        mapping["key2.key3.key4"] = "value2"
        assert mapping["key2.key3.key4"] == "value2"
        with pytest.raises(TypeError):
            mapping["key1.key2.key3"] = "value3"

    def test_del_item(self):
        mapping = NestedDict()
        mapping["key1"] = "value1"
        del mapping["key1"]
        with pytest.raises(KeyError):
            _ = mapping["key1"]

        mapping["key2.key3.key4"] = "value2"
        del mapping["key2.key3.key4"]
        with pytest.raises(KeyError):
            _ = mapping["key2.key3.key4"]
        assert mapping["key2.key3"] == {}

    def test_iter(self):
        mapping = NestedDict()
        mapping["key1"] = "value1"
        mapping["key2.key3.key4"] = "value2"
        assert list(mapping.keys()) == ["key1", "key2"]

    def test_len(self):
        mapping = NestedDict()
        mapping["key1"] = "value1"
        mapping["key2.key3.key4"] = "value2"
        assert len(mapping) == 2

    def test_overwrite_item(self):
        mapping = NestedDict()
        mapping["key1"] = "value1"
        mapping["key1"] = "value2"
        assert mapping["key1"] == "value2"

        mapping = NestedDict()
        mapping["key1.key2.key3"] = "value1"
        mapping["key1.key2.key3"] = "value2"
        assert mapping["key1.key2.key3"] == "value2"

        mapping = NestedDict()
        mapping["key1.key2.key3"] = "value1"
        mapping["key1.key2"] = "value2"
        assert mapping["key1.key2"] == "value2"


class TestFramedArray:
    def test_iloc(self):
        kedges = np.arange(0.0, 0.20 + 0.01 / 2, 0.01)
        ii = ArrayEdges(["pre", "post", "cross"])
        il = ArrayEdges([0, 2, 4])
        ik = ArrayEdges(kedges[:-1], kedges[1:])
        fa = FramedArray(np.random.random((3, 3, 20)), edges={"i": ii, "l": il, "k": ik})

        def test(value):
            np.testing.assert_array_equal(value, exp)

        exp = fa.value[:]
        test(fa[:].value)
        test(fa.iloc[:].value)

        exp = fa.value[...]
        test(fa[...].value)
        test(fa.iloc[...].value)

        exp = fa.value[0, 2, 11]
        test(fa[0, 2, 11].value)
        test(fa.iloc[0, 2, 11].value)

        exp = fa.value[1, :]
        test(fa[1, :].value)
        test(fa.iloc[1, :].value)

        exp = fa.value[1, ...]
        test(fa[1, ...].value)
        test(fa.iloc[1, ...].value)

        exp = fa.value[:, -2, :]
        test(fa[:, -2, :].value)
        test(fa.iloc[:, -2, :].value)

        exp = fa.value[-2, ..., -7]
        test(fa[-2, ..., -7].value)
        test(fa.iloc[-2, ..., -7].value)

        exp = fa.value[1::2, ..., -7:-1:2]
        test(fa[1::2, ..., -7:-1:2].value)
        test(fa.iloc[1::2, ..., -7:-1:2].value)

        with xfail:
            exp = fa.value[:, [2], [11]]
            test(fa[:, [2], [11]].value)
            test(fa.iloc[:, [2], [11]].value)

    def test_loc(self):
        kedges = np.arange(0.0, 0.20 + 0.01 / 2, 0.01)
        ii = ArrayEdges(["pre", "post", "cross"])
        il = ArrayEdges([0, 2, 4])
        ik = ArrayEdges(kedges[:-1], kedges[1:])
        fa = FramedArray(np.random.random((3, 3, 20)), edges={"i": ii, "l": il, "k": ik})

        def test(value):
            np.testing.assert_array_equal(value, exp)

        exp = fa[:].value
        test(fa.loc[...].value)
        test(fa.loc[:].value)

        exp = fa[0, 0].value
        test(fa.sel(l=0, i="pre").value)
        exp = fa[1, 2].value
        test(fa.sel(l=4, i="post").value)
        exp = fa[2, [1]].value
        test(fa.sel(l=[2], i="cross").value)
        exp = fa[1:, :2].value
        test(fa.loc["post":, :2].value)

        exp = fa[[2], 1:, 10:15].value
        test(fa.sel(i=["cross"], l=slice(2, 4), k=slice(0.1, 0.15)).value)

        exp = fa[[2], 1:, 10].value
        test(fa.sel(i=["cross"], l=slice(2, 4), k=0.11).value)

        exp = fa[:, :, 10].value
        test(fa.sel(i=slice(None), l=slice(None), k=0.11).value)

    def test_save_and_load(self, tmp_path):
        kedges = np.arange(0.0, 0.20 + 0.01 / 2, 0.01)
        ii = ArrayEdges(["pre", "post", "cross"])
        il = ArrayEdges([0, 2, 4])
        ik = ArrayEdges(kedges[:-1], kedges[1:])
        fa = FramedArray(np.random.random((3, 3, 20)), edges={"i": ii, "l": il, "k": ik})
        filename = tmp_path / "test.pickle"
        fa.save(filename)
        fa2 = FramedArray.load(filename)
        np.testing.assert_allclose(fa.value, fa2.value)
