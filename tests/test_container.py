from __future__ import annotations
import pytest
from lsstools.container import NestedDict


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
