import pytest
from lsstools.ex.dataset import DAGGraph, Option, Dataset, NodeEvaluationError, PatternMatchingNode


class TestOption:
    def test_init(self):
        op = Option(name="test")
        assert op.name == "test"
        assert op.default is op.MISSING
        assert op.choices is op.ANY
        assert op.type is str

        op = Option(name="test", default=3)
        assert op.default == 3
        assert op.choices is op.ANY
        assert op.type is int

        class MyInt(int):
            pass

        op = Option(name="test", choices=[None, 1, MyInt(2), 3])
        assert op.default is op.MISSING
        assert op.choices == {None, 1, 2, 3}
        assert op.type is int

        op = Option(name="test", default=None, choices=[None, 1, MyInt(2), 3], type=MyInt)
        assert op.default is None
        assert op.choices == {None, MyInt(1), MyInt(2), MyInt(3)}
        assert op.type is MyInt

        with pytest.raises(ValueError):
            op = Option(name="test", default=1.1, choices=[None, 1, MyInt(2), 3])

    def test_str_behaviour(self):
        op = Option(name="los", default="z", choices=[None, "x", "y", "z"])
        assert str(op) == "los"

        op2 = op + op
        assert op2.name == op.name + op.name
        assert op2.default is op2.MISSING
        assert op2.choices is op2.ANY
        assert op2.type is str

        op2 = op + "los"
        assert op2.name == op.name + op.name
        assert op2.default == op.default
        assert op2.choices == op.choices
        assert op2.type == op.type

        from copy import copy, deepcopy

        op2 = copy(op)
        assert op2.name == op.name
        assert op2.default == op.default
        assert op2.choices == op.choices
        assert op2.type == op.type

        op2 = deepcopy(op)
        assert op2.name == op.name
        assert op2.default == op.default
        assert op2.choices == op.choices
        assert op2.type == op.type


def get_graph():
    graph = {
        "b": {"a"},
        "c": {"a"},
        "d": {"a", "b", "c"},
        "e": {"d", "c", "aa"},
        "f": {"a", "c", "aa"},
    }
    return DAGGraph(graph)


class TestDAGGraph:
    def test_inputs(self):
        g = get_graph()
        assert g.inputs() == ["a", "aa"]

    def test_ancestors(self):
        g = get_graph()
        results = {
            "a": set(),
            "aa": set(),
            "b": {"a"},
            "c": {"a"},
            "d": {"a", "b", "c"},
            "e": {"a", "b", "c", "d", "aa"},
            "f": {"a", "c", "aa"},
        }
        for k, ref in results.items():
            assert g.ancestors(k) == ref

    def test_descendants(self):
        g = get_graph()
        results = {
            "a": {"b", "c", "d", "e", "f"},
            "aa": {"e", "f"},
            "b": {"d", "e"},
            "c": {"d", "e", "f"},
            "d": {"e"},
            "e": set(),
            "f": set(),
        }
        for k, ref in results.items():
            assert g.descendants(k) == ref


def test_pattern_matching_node():
    node = PatternMatchingNode(
        name="test",
        patterns=[
            ("'BGS' in tracer", "{tracer}"),
            ("'LRG' in tracer", "LRG"),
            ("tracer == 'ELG2' and redshift == '0.8_1.1'", 2),
            ("'QSO' in tracer", "!{lambda tracer: tracer.lower()}!"),
            ("tracer == 'LBG'", "{tracer} {redshift}"),
            ("_", "unknown"),
        ],
    )
    assert node.name == "test"
    assert node.depends == ["tracer", "redshift"]
    assert node(tracer="BGS_BRIGHT-21.5") == "BGS_BRIGHT-21.5"
    assert node(tracer="LRG1") == "LRG"
    assert node(tracer="QSO2") == "qso2"
    assert node(tracer="ELG2", redshift="0.8_1.1") == 2
    assert node(tracer="LBG", redshift="2.1_3.0") == "LBG 2.1_3.0"
    assert node(tracer="LAE") == "unknown"


def test_dataset():
    from decimal import Decimal as D

    def get_tracer_name(bin_name):
        if "BGS" in bin_name:
            return "BGS_BRIGHT-21.5"
        if "LRG" in bin_name:
            return "LRG"
        if "ELG" in bin_name:
            return "ELG"
        if "QSO" in bin_name:
            return "QSO"
        raise NotImplementedError

    ds = Dataset(
        options=[
            "bin_name",
            Option(name="observable", default="power", choices=["power", "corr"]),
            "boxsize",
        ],
        derived={
            "volume": lambda boxsize: boxsize**3,
            "zmin": {
                "bin_name": {
                    "BGS": D("0.1"),
                    "LRG1": D("0.4"),
                    "LRG2": D("0.6"),
                    "LRG3": D("0.8"),
                    "ELG2": D("1.1"),
                    "QSO": D("0.8"),
                }
            },
            "zmax": {
                "bin_name": {
                    "BGS": D("0.4"),
                    "LRG1": D("0.6"),
                    "LRG2": D("0.8"),
                    "LRG3": D("1.1"),
                    "ELG2": D("1.6"),
                    "QSO": D("2.1"),
                }
            },
            "redshift": "{zmin}_{zmax}",
            "tracer": get_tracer_name,
            "path": "/desi/pk/{observable}_{tracer}_z{redshift}_boxsize{boxsize}.npy",
        },
    )


def test_extended_evaluation_rule():
    ds = Dataset(
        options=["A", "B", "C"],
        derived={"D": "{A} {B}", "E": "{C}", "F": "{D} {B} {E}", "G": "{F}"},
    )
    assert ds.value("F", A="A", B="B", C="C") == "A B B C"
    assert ds.value("F", B="B", C="C", D="D") == "D B C"
    assert ds.value("F", B="B", D="D", E="E") == "D B E"
    assert ds.value("G", F="F") == "F"
    assert ds.value("G", G="G") == "G"
    with pytest.raises(NodeEvaluationError):
        ds.value("F", D="D", E="E")


def test_dataset_iter():
    ds = Dataset(
        options=[Option("A", default="a"), Option("B", choices=["b1", "b2", "b3"]), "C"],
        derived={"D": "{A} {B}", "E": "{C}", "F": "{D} {B} {E}", "G": "{F}"},
    )
    with pytest.raises(NodeEvaluationError):
        set(ds.iter("F"))
    # default
    assert set(ds.iter("F", C="C")) == set(["a b1 b1 C", "a b2 b2 C", "a b3 b3 C"])
    # replaceholder rule
    assert set(ds.iter("F", C=["c1", "c2"], B=...)) == set(
        ["a b1 b1 c1", "a b2 b2 c1", "a b3 b3 c1", "a b1 b1 c2", "a b2 b2 c2", "a b3 b3 c2"]
    )
    assert ds.iter_list("F", C=["c1", "c2"], B=...) == ds.iter_list("F", C=["c1", "c2"])
    assert ds.iter_list("F", C=["c1", "c2"], B=...) != ds.iter_list("F", B=..., C=["c1", "c2"])
    # extended evaluation rule
    assert ds.iter_list("F", F="F") == ["F"]
    assert set(ds.iter("G", F=["f1", "f2", "f3"])) == set(["f1", "f2", "f3"])
    assert set(ds.iter("F", D="D", C=["c1", "c2"])) == set(
        ["D b1 c1", "D b2 c1", "D b3 c1", "D b1 c2", "D b2 c2", "D b3 c2"]
    )
    assert set(ds.iter("F", D="D", E="E")) == set(["D b1 E", "D b2 E", "D b3 E"])


if __name__ == "__main__":
    test_dataset_iter()
