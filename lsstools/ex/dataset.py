import ast
import enum
import graphlib
import numpy as np
import re
import yaml
from collections import deque, UserString, UserDict
from decimal import Decimal
from ..utils import format_collection
from ..static_typing import *


import yaml.composer
import yaml.scanner

_built_in_type = type
_T = TypeVar("_T")
_S = TypeVar("_S")


class Resolver(yaml.resolver.BaseResolver):
    pass


Resolver.add_implicit_resolver(  # regex copied from yaml source
    "!decimal",
    re.compile(
        r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-9]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN)
    )$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)

for ch, vs in yaml.resolver.Resolver.yaml_implicit_resolvers.items():
    Resolver.yaml_implicit_resolvers.setdefault(ch, []).extend(
        (tag, regexp) for tag, regexp in vs if not tag.endswith("float")
    )


class Loader(
    yaml.reader.Reader,
    yaml.scanner.Scanner,
    yaml.parser.Parser,  # type: ignore
    yaml.composer.Composer,
    yaml.constructor.SafeConstructor,
    Resolver,
):
    def __init__(self, stream):
        yaml.reader.Reader.__init__(self, stream)
        yaml.scanner.Scanner.__init__(self)
        yaml.parser.Parser.__init__(self)  # type: ignore
        yaml.composer.Composer.__init__(self)
        yaml.constructor.SafeConstructor.__init__(self)
        Resolver.__init__(self)


def decimal_constructor(loader, node):
    value = loader.construct_scalar(node)
    return Decimal(value)


yaml.add_constructor("!decimal", decimal_constructor, Loader)


def yaml_load(stream, **kwargs):
    return yaml.load(stream, Loader, **kwargs)


def cast_type(obj, type: type[_T], ignore_none=True) -> _T:
    if isinstance(obj, type):
        return obj
    if obj is None and ignore_none:
        return obj
    return type(obj)  # type: ignore


def common_type(types: Sequence[type]) -> type:
    if not types:
        return object
    common = types[0]
    for typ in types:
        if issubclass(common, typ):
            common = typ
        elif issubclass(typ, common):
            common = common
        else:
            return object
    return common


def is_magic_expr(expr: str) -> bool:
    if expr.startswith("!{") and expr.endswith("}!"):
        return True
    return False


def parse_range_expr(expr: str) -> list[int]:
    """
    Examples
    --------
    1..5    => [1, 2, 3, 4, 5]
    1..5..2 => [1, 3, 5]
    """
    try:
        match expr.count(".."):
            case 1:
                (start, stop), step = map(int, expr.split("..")), 1
            case 2:
                start, stop, step = map(int, expr.split(".."))
            case _:
                raise ValueError("invalid range expr")
    except TypeError:
        raise ValueError("invalid range expr")
    return list(range(start, stop + 1, step))


def import_object(name: str) -> Any:
    """import object from string, e.g. "numpy:exp", "module_name:class_name.class_method"""
    import importlib
    import functools

    module_name, object_name = name.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return functools.reduce(getattr, object_name.split("."), module)


def parse_magic_expr(
    expr: str,
    rules: Container[Literal["range", "lambda", "dyobject"]] = {"range", "lambda", "dyobject"},
) -> Any:
    if not is_magic_expr(expr):
        raise ValueError("not a valid magic expr")
    expr = expr[2:-2]
    if "range" in rules and ".." in expr:
        return parse_range_expr(expr)
    if "lambda" in rules and expr.startswith("lambda"):
        return eval(expr)
    if "dyobject" in rules and ":" in expr:
        return import_object(expr)
    raise ValueError(f"invalid magic expr !{{{expr}}}!")


def parse_type(expr: str):
    from decimal import Decimal

    if is_magic_expr(expr):
        return parse_magic_expr(expr, rules={"dyobject"})
    _globals = {"Decimal": Decimal}
    return eval(expr, _globals)


class OptionSentinel(enum.Enum):
    ANY = "ANY"
    INFERRED = "INFERRED"
    MISSING = "MISSING"


class Option(UserString):
    MISSING = OptionSentinel.MISSING
    ANY = OptionSentinel.ANY
    INFERRED = OptionSentinel.INFERRED

    def __init__(
        self,
        name: str,
        default: Any | Literal[OptionSentinel.MISSING] = OptionSentinel.MISSING,
        choices: Collection[None | Any] | Literal[OptionSentinel.ANY] = OptionSentinel.ANY,
        type: type | Literal[OptionSentinel.INFERRED] = OptionSentinel.INFERRED,
    ) -> None:
        super().__init__(name)
        if default is self.MISSING and choices is self.ANY and type is self.INFERRED:
            # default type: str
            type = str
        self._default = default
        self._choices = set(choices) if (choices and choices is not self.ANY) else self.ANY
        if type is self.INFERRED:
            # infer type from default or choices, ignoring Sentinel and None
            types = set()
            if self._default not in (None, self.MISSING):
                types |= {_built_in_type(self._default)}
            if self._choices is not self.ANY:
                for x in self._choices:
                    if x is not None:
                        types.add(_built_in_type(x))
            _type = common_type(list(types))
            if _type is object:
                objects = set()
                if self._default is not self.MISSING:
                    objects |= {self._default}
                if self._choices is not self.ANY:
                    objects |= set(self._choices)
                raise ValueError(f"{objects} do not have common type")
            self._type = _type
        else:
            self._type = type
        if self._default is not self.MISSING:
            self._default = cast_type(self._default, self._type, ignore_none=True)
        self._choices = (
            set(cast_type(_, self._type, ignore_none=True) for _ in self._choices)
            if (self._choices is not self.ANY)
            else self.ANY
        )
        self._frozen_choices = (
            frozenset(self._choices) if (self._choices is not self.ANY) else self.ANY
        )
        if self._choices is not self.ANY and self._default is not self.MISSING:
            if self._default not in self._choices:
                raise ValueError(f"{self._default} not in {self._choices}")

    @property
    def name(self):
        return self.data

    @property
    def default(self) -> Literal[OptionSentinel.MISSING] | Any:
        return self._default

    @property
    def choices(self) -> Literal[OptionSentinel.ANY] | frozenset[Any]:
        return self._frozen_choices  # type: ignore

    @property
    def type(self):
        return self._type

    def bounded(self) -> bool:
        return self.choices is not self.ANY

    def has_default(self) -> bool:
        return self.default is not self.MISSING

    def clone(self, **kwargs):
        x: dict = dict(name=self.name, default=self.default, choices=self.choices, type=self.type)
        x |= kwargs
        return type(self)(**x)

    def info(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name}, "
            f"default={self.default if self.default is not self.MISSING else 'MISSING'}, "
            f"choices={format_collection(self.choices) if self.choices is not self.ANY else 'ANY'}, "
            f"type={self.type.__name__}"
            f")"
        )

    def __copy__(self):
        return self.clone()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, type={self.type.__name__})"

    @classmethod
    def from_dict(cls, name: str, spec: Any | dict[str, Any] = {}, **kwargs):
        if not isinstance(spec, dict):
            spec = {"default": spec}
        spec |= kwargs
        for k, v in spec.items():
            if isinstance(v, str) and is_magic_expr(v):
                if k == "default":
                    spec[k] = parse_magic_expr(v, rules={"range", "dyobject"})
                if k == "choices":
                    spec[k] = parse_magic_expr(v, rules={"range", "dyobject"})
                if k == "type":
                    spec[k] = parse_type(v)
        return cls(name, **spec)

    # override methods inherited from str
    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, str):
            return self.clone(name=self.data + other)
        return self.clone(name=self.data + str(other))

    def __radd__(self, other):
        if isinstance(other, str):
            return self.clone(name=other + self.data)
        return self.clone(name=str(other) + self.data)

    def __mul__(self, n):
        return self.clone(name=self.data * n)

    __rmul__ = __mul__

    def __mod__(self, args):
        return self.clone(name=self.data % args)

    def __rmod__(self, template):
        return self.clone(name=str(template) % self)

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.clone(name=self.data.capitalize())

    def casefold(self):
        return self.clone(name=self.data.casefold())

    def center(self, width, *args):
        return self.clone(name=self.data.center(width, *args))

    def removeprefix(self, prefix, /):
        if isinstance(prefix, UserString):
            prefix = prefix.data
        return self.clone(name=self.data.removeprefix(prefix))

    def removesuffix(self, suffix, /):
        if isinstance(suffix, UserString):
            suffix = suffix.data
        return self.clone(name=self.data.removesuffix(suffix))

    def expandtabs(self, tabsize=8):
        return self.clone(name=self.data.expandtabs(tabsize))

    def join(self, seq):
        return self.clone(name=self.data.join(seq))

    def ljust(self, width, *args):
        return self.clone(name=self.data.ljust(width, *args))

    def lower(self):
        return self.clone(name=self.data.lower())

    def lstrip(self, chars=None):
        return self.clone(name=self.data.lstrip(chars))

    def replace(self, old, new, maxsplit=-1):
        if isinstance(old, UserString):
            old = old.data
        if isinstance(new, UserString):
            new = new.data
        return self.clone(name=self.data.replace(old, new, maxsplit))

    def rjust(self, width, *args):
        return self.clone(name=self.data.rjust(width, *args))

    def rstrip(self, chars=None):
        return self.clone(name=self.data.rstrip(chars))

    def strip(self, chars=None):
        return self.clone(name=self.data.strip(chars))

    def swapcase(self):
        return self.clone(name=self.data.swapcase())

    def title(self):
        return self.clone(name=self.data.title())

    def translate(self, *args):
        return self.clone(name=self.data.translate(*args))

    def upper(self):
        return self.clone(name=self.data.upper())

    def zfill(self, width):
        return self.clone(name=self.data.zfill(width))


class DAGGraph(UserDict[_T, set[_T]]):
    """Directed acyclic graph."""

    def __init__(self, graph: dict[_T, set[_T]] | None = None):
        self.data = {}
        if graph is not None:
            self.data.update(graph)
        # keep free nodes in the graph
        self.data = {node: set() for node in self._all_nodes() if node not in self.data} | self.data
        self._validate()

    def copy(self):
        # shallow copy, node itself will not be copied
        return type(self)({k: set(v) for k, v in self.items()})

    def empty(self) -> bool:
        return bool(self.data)

    def inputs(self) -> list[_T]:
        return [node for node, depends in self.items() if not depends]

    def size(self) -> int:
        return len(self)

    def clear_orphans(self):
        for node in self.orphans():
            del self.data[node]

    def orphans(self) -> list[_T]:
        retval = []
        for node in self.inputs():
            if not self.descendants(node):
                retval.append(node)
        return retval

    def neighbours(self, node: _T) -> set[_T]:
        return self[node]

    def ancestors(self, node: _T) -> set[_T]:
        if node not in self.keys():
            raise ValueError(f"{node} not in the graph")
        frontier = deque(self[node])  # wait for predecessor checking
        retval = dict.fromkeys(self[node])  # visited
        while frontier:
            item = frontier.popleft()
            for predecessor in self.get(item, []):
                if predecessor not in retval.keys():
                    frontier.append(predecessor)
                    retval[predecessor] = None
        return set(retval.keys())

    def descendants(self, node: _T, indirect=True) -> set[_T]:
        if node not in self.keys():
            raise ValueError(f"{node} not in the graph")
        if not indirect:
            return set(k for k, v in self.items() if node in v)
        frontier = deque([node])
        retval = set()
        while frontier:
            item = frontier.popleft()
            for k, v in self.items():
                if k in retval:
                    continue
                if item in v and k not in retval:
                    frontier.append(k)
                    retval.add(k)
        return retval

    def subgraph(self, *, select: Iterable[_T] | None = None, delete: Iterable[_T] | None = None):
        if select is None and delete is None:
            return self.copy()
        if select is not None and delete is not None:
            raise ValueError(f"`select` and `delete` are exclusive")
        # retain the node type
        mapping = {k: k for k in self.keys()}
        if select is not None:
            try:
                select = [mapping[k] for k in select]
            except KeyError:
                raise ValueError(f"{select} not in the graph")
            selected = set(select)
            for node in select:
                selected |= self.ancestors(node)
            return type(self)({node: set(self[node]) for node in selected})
        if delete is not None:
            try:
                delete = [mapping[k] for k in delete]
            except KeyError:
                raise ValueError(f"{delete} not in the graph")
            deleted = set()
            for node in deleted:
                deleted |= self.descendants(node)
            return type(self)(
                {node: set(depends) for node, depends in self.items() if node not in deleted}
            )
        raise AssertionError("Expected code to be unreachable")

    def __setitem__(self, key: _T, item: set[_T]) -> None:
        super().__setitem__(key, item)
        for node in item:
            if node not in self:
                self.data[node] = set()
        self._validate()

    def __delitem__(self, key: _T) -> None:
        try:
            succ = self.descendants(key)
        except ValueError:
            raise KeyError(f"{key}")
        del self.data[key]
        for node in succ:
            del self.data[node]

    def _all_nodes(self) -> list[_T]:
        all_nodes = dict()
        for node, depends in self.items():
            all_nodes[node] = None
            all_nodes |= dict.fromkeys(depends)
        return list(all_nodes.keys())

    def _validate(self):
        graphlib.TopologicalSorter(self.data).prepare()


class NodeEvaluationError(ValueError):
    pass


@runtime_checkable
class FunctorNode(Protocol):
    """Named callable node."""

    name: str
    depends: list[str]

    def __call__(self, **kwargs): ...
    def allows_override(self, value) -> bool: ...
    def type_cast(self, value) -> Any: ...


class FunctorGraph(DAGGraph[FunctorNode]):
    """DAG graph with named callable node."""

    def node(self, key):
        mapping = {_: _ for _ in self.keys()}
        return mapping[key]

    def value(self, _name: Any, /, *, ignore_redundant=True, **option_spec):
        """Evaluate the node w/ input option_specs.

        If `ignore_redundant` is false, this method will raise a `ValueError` if `option_spec` has any redundant
        options. Otherwise, redundant options will be ignored. In addition, values provided by `option_spec` could
        participate in the graph evaluation depending on whether the value of `FunctorNode` allows to be overridden.
        """
        # XXX: cache
        if _name not in self:
            raise KeyError(f"`{_name}` not in the graph")
        for k in option_spec.keys():
            if k not in self:
                raise KeyError(f"`{k}` not in graph")
        g = self.subgraph(select=[_name])
        if redundant := set(option_spec.keys()).difference(g.inputs()):
            if not ignore_redundant:
                raise ValueError(f"redundant options: {redundant}")
            # override
            for x in redundant:
                if x in g:
                    g.data[self.node(x)] = set()
            g = g.subgraph(select=[_name])
        _missing = object()
        results: dict[str, Any] = {}
        node: FunctorNode
        for node in graphlib.TopologicalSorter(g.data).static_order():  # type: ignore
            override = option_spec.get(node.name, _missing)
            if override is _missing:
                results[node.name] = node(**{depend: results[depend] for depend in node.depends})
            else:
                # automatic type conversion
                override = node.type_cast(override)
                if not node.allows_override(override):
                    raise NodeEvaluationError(
                        f"node `{node}` does not allow to have a value of `{override}`"
                    )
                results[node.name] = override
        return results[str(_name)]

    def take(
        self, _name: Any, _default: _S = None, /, *, ignore_redundant=True, **option_spec
    ) -> _S:
        try:
            return self.value(_name, ignore_redundant=ignore_redundant, **option_spec)
        except NodeEvaluationError:
            return _default


class BaseFunctorNode(FunctorNode):
    def __call__(self, **kwargs):
        raise NotImplementedError

    def allows_override(self, value) -> bool:
        return True

    def type_cast(self, value) -> Any:
        return value

    def __eq__(self, other):
        if isinstance(other, BaseFunctorNode):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"

    def __str__(self):
        return self.name


class OptionNode(BaseFunctorNode):
    def __init__(self, option: str | Option) -> None:
        if not isinstance(option, Option):
            option = Option(name=str(option))
        self.option = option

    @property
    def name(self):
        return self.option.name

    @property
    def depends(self):
        return []

    def __call__(self):
        if not self.option.has_default():
            raise NodeEvaluationError(f"node `{self}` does not have default value")
        return self.option.default

    def allows_override(self, value) -> bool:
        if not self.option.bounded():
            return True
        return cast_type(value, self.option.type, ignore_none=True) in self.option.choices  # type: ignore

    def type_cast(self, value):
        return cast_type(value, self.option.type, ignore_none=True)


class FStringNode(BaseFunctorNode):
    def __init__(self, name, s: str) -> None:
        self.name = name
        self.s = s

    @property
    def depends(self):
        import string

        return [
            field_name for _, field_name, _, _ in string.Formatter().parse(self.s) if field_name
        ]

    def __call__(self, **kwargs):
        return self.s.format(**kwargs)


class CallableNode(BaseFunctorNode):
    def __init__(self, name, fn: Callable) -> None:
        self.name = name
        self.fn = fn
        _ = self.depends

    @property
    def depends(self):
        import inspect

        retval = []
        sig = inspect.signature(self.fn)
        for param in sig.parameters.values():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise ValueError(f"unsupported callable {self.fn}")
            retval.append(param.name)
        return retval

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class SingleMappingNode(BaseFunctorNode):
    def __init__(self, name, mapping: Mapping[str, dict]) -> None:
        if len(mapping) != 1:
            raise ValueError(f"not a single mapping")
        self.name = name
        self._mapping = mapping

    @property
    def depends(self):
        return list(self._mapping.keys())

    def __call__(self, **kwargs):
        name = self.depends[0]
        value = kwargs[name]
        return self._mapping[name][value]


class VariableNameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variable_names = set()

    def visit_Name(self, node: ast.Name):
        self.variable_names.add(node.id)
        self.generic_visit(node)


def extract_variable_names(expression, env: set = set()):
    tree = ast.parse(expression)
    visitor = VariableNameVisitor()
    visitor.visit(tree)
    return visitor.variable_names - env


class _Value:
    def __init__(self, value) -> None:
        self.value = value

    def __call__(self):
        return self.value


class PatternMatchingNode(BaseFunctorNode):
    # XXX: PEG for complicated rules
    def __init__(self, name, patterns: Sequence[tuple[str, Any]]) -> None:
        self.name = name
        self.depends = []
        self.patterns: list[tuple[str, FunctorNode]] = []
        for i, (pattern, action) in enumerate(patterns):
            self.depends.extend(extract_variable_names(pattern, env={"Decimal", "np"}))
            if isinstance(action, (int, float, Decimal)):
                # pure value, cannot use lambda here
                action = CallableNode(name=str(i), fn=_Value(action))
            elif isinstance(action, str):
                if is_magic_expr(action):
                    # magic value or callable
                    action = parse_magic_expr(action, rules=["lambda", "dyobject"])
                    action = CallableNode(
                        name=str(i), fn=action if isinstance(action, Callable) else _Value(action)
                    )
                else:
                    action = FStringNode(name=str(i), s=action)
            elif isinstance(action, FunctorNode):
                action = action
            elif isinstance(action, Callable):
                action = CallableNode(name=str(i), fn=action)
            else:
                raise TypeError(f"unexpected element {action}")
            self.depends.extend(action.depends)
            self.patterns.append((pattern, action))
        self.depends = list(dict.fromkeys(self.depends))

        def wildcard_fn():
            raise ValueError(f"all matching failed")

        self.wildcard = CallableNode(name="wildcard", fn=wildcard_fn)
        for i, (pattern, action) in enumerate(self.patterns):
            if pattern == "_":
                if i != len(self.patterns) - 1:
                    raise ValueError(f"wildcard makes remaining patterns unreachable")
                _, self.wildcard = self.patterns.pop()
                self.depends.remove("_")

    def __call__(self, **kwargs):
        env = {"np": np, "Decimal": Decimal} | kwargs
        for pattern, action in self.patterns:
            if eval(pattern, env):
                return action(**{k: kwargs[k] for k in action.depends})
        return self.wildcard(**{k: kwargs[k] for k in self.wildcard.depends})


def convert_to_functor_nodes(
    derived: dict[str, str | Callable | dict | FunctorNode] | Sequence[FunctorNode]
):
    _derived: dict[str, FunctorNode] = {}
    if isinstance(derived, dict):
        for k, v in derived.items():
            if isinstance(v, FunctorNode):
                _derived[k] = v
            elif isinstance(v, str):
                _derived[k] = FStringNode(name=k, s=v)
            elif isinstance(v, Callable):
                _derived[k] = CallableNode(name=k, fn=v)
            elif isinstance(v, dict):
                _derived[k] = SingleMappingNode(name=k, mapping=v)
            elif isinstance(v, list):
                _derived[k] = PatternMatchingNode(name=k, patterns=v)
            else:
                raise TypeError(f"unexpected element {v}")
    else:
        _derived = {v.name: v for v in derived}
    return _derived


class Dataset:
    def __init__(
        self,
        options: Sequence[str | Option | OptionNode],
        derived: dict[str, str | Callable | dict | FunctorNode] | Sequence[FunctorNode],
    ):
        self._options: list[Option] = []
        for op in options:
            if isinstance(op, OptionNode):
                self._options.append(op.option)
            elif isinstance(op, Option):
                self._options.append(op)
            else:
                self._options.append(Option(name=str(op)))
        self._derived = convert_to_functor_nodes(derived)

        graph: dict = {OptionNode(_): set() for _ in self._options}
        graph |= {_: set(_.depends) for _ in self._derived.values()}
        mapping = {k: k for k in graph.keys()}
        try:
            graph = {k: set(mapping[_] for _ in v) for k, v in graph.items()}
        except KeyError as ex:
            raise ValueError(f"options do not cover all inputs") from ex
        self._graph = FunctorGraph(graph=graph)

    @property
    def graph(self):
        return self._graph

    @property
    def required_options(self):
        return [
            option.name for option in self.options(to_str=False) if option.default is option.MISSING
        ]

    def options(self, to_str=True) -> Any:
        if to_str:
            return [_.name for _ in self._options]
        return self._options

    def derived(self, to_str=True) -> Any:
        if to_str:
            return [_ for _ in self._derived.keys()]
        return list(self._derived.values())

    def choices(self, node: str):
        if node not in self.options():
            return frozenset()
        [option] = [_ for _ in self.options(to_str=False) if _.name == node]
        return option.choices

    def depends(self, node, all_options=True):
        if all_options:
            c = self.options()
        else:
            c = self.required_options
        return [node.name for node in self.graph.ancestors(node) if node.name in c]

    def select(self, *, inplace=False, ignore_redundant=True, **option_spec):
        """narrowing"""
        if not ignore_redundant:
            if redundant := set(option_spec.keys()).difference(self.options()):
                raise ValueError(f"redundant options: {redundant}")
        options = []
        _missing = object()
        for node in self.options(to_str=False):
            _update: dict = {}
            if (v := option_spec.get(node.name, _missing)) is not _missing:
                if isinstance(v, dict):
                    _update.update(v)
                elif isinstance(v, (tuple, list)):
                    _update = dict(choices=v)
                    if node.default not in {cast_type(_, node.type, ignore_none=True) for _ in v}:
                        _update["default"] = Option.MISSING
                else:
                    _update = dict(default=v)
            options.append(node.clone(**_update))
        derived: dict = {k: v for k, v in self._derived.items()}
        ds = type(self)(options=options, derived=derived)
        if not inplace:
            return ds
        self._options = ds._options
        self._derived = ds._derived
        self._graph = ds._graph
        return self

    def iter(self, _name: Any, /, *, ignore_redundant=True, **option_spec):
        """Iterate and evaluate the graph.

        Parameters
        ----------
        _name : Any
            Node name.
        ignore_redundant : bool, optional
            Wheter to ignore redundant option_spec, by default True.

        Notes
        -----
        This method will keep the ordering specified by option_spec. One can use ... or [...]
        as the placeholder to take default value or choices.
        """
        import itertools

        # typo check
        for k in option_spec:
            if k not in self.graph:
                raise KeyError(k)
        # redundant check
        g = self.graph.subgraph(select=[_name])
        if redundant := set(option_spec.keys()).difference(_.name for _ in g):
            if not ignore_redundant:
                raise ValueError(f"redundant options: {redundant}")
            # 1st filtering
            option_spec = {k: v for k, v in option_spec.items() if k not in redundant}
        if derived := set(option_spec.keys()).difference(_.name for _ in g.inputs()):
            if not ignore_redundant:
                raise ValueError(f"redundant options: {derived}")
            # 2nd filtering
            for d in derived:
                g[g.node(d)] = set()
            g = g.subgraph(select=[_name])
        # regularize input
        param_values_dict: dict[str, Iterable] = {}
        for option_name, value in option_spec.items():
            if value != ... and value != [...]:
                # make list
                param_values_dict[option_name] = (
                    value if isinstance(value, (tuple, list)) else [value]
                )
            else:
                # placeholder ... or [...]
                input_options_mapping: dict[str, Option] = {
                    _.name: _ for _ in self.options(to_str=False)
                }
                if option_name not in input_options_mapping:
                    # derived
                    raise ValueError(f"derived node cannot use {value} as the placeholder")
                option = input_options_mapping[option_name]
                if value is ...:
                    if option.default is option.MISSING:
                        raise ValueError(f"option {option} does not have a default value")
                    param_values_dict[option_name] = [option.default]
                else:
                    if option.choices is option.ANY:
                        raise ValueError(f"option {option} is not bounded")
                    param_values_dict[option_name] = option.choices
        # inherit default option value and choices if not specified
        for input_node in g.inputs():
            if input_node.name not in param_values_dict:
                option = {_.name: _ for _ in self.options(to_str=False)}[input_node.name]
                if option.default is option.MISSING:
                    if option.choices is option.ANY:
                        # not used or have derived in the option_spec
                        continue
                    param_values_dict[input_node.name] = option.choices
                else:
                    param_values_dict[input_node.name] = [option.default]

        for values in itertools.product(*param_values_dict.values()):
            input_dict = dict(zip(param_values_dict.keys(), values))
            yield self.value(_name, ignore_redundant=ignore_redundant, **input_dict)

    def iter_list(self, _name: Any, /, *, ignore_redundant=True, **option_spec):
        return list(self.iter(_name, ignore_redundant=ignore_redundant, **option_spec))

    def value(self, _name: Any, /, *, ignore_redundant=True, **option_spec):
        return self.graph.value(_name, ignore_redundant=ignore_redundant, **option_spec)

    def take(
        self, _name: Any, _default: _S = None, /, *, ignore_redundant=True, **option_spec
    ) -> _S:
        return self.graph.take(_name, _default, ignore_redundant=ignore_redundant, **option_spec)

    def graph_repr(self):
        return {k.name: set(_.name for _ in v) for k, v in self.graph.data.items()}


class DatasetCollection(UserDict[str, Dataset]):
    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]):
        dc = {}
        alias = {}
        for name, v in input_dict.items():
            options = [
                Option.from_dict(name=option_name, spec=option_spec)
                for option_name, option_spec in v["options"].items()
            ]
            derived = {
                derived_name: derived_spec for derived_name, derived_spec in v["derived"].items()
            }
            if _alias := v.get("alias"):
                alias[name] = _alias
            dc[name] = Dataset(options=options, derived=derived)
        for _from, _to in alias.items():
            dc[_to] = dc[_from]
        return cls(dc)

    @classmethod
    def load_json(cls, path=None):
        from pathlib import Path

        if path is None:
            path = Path(__file__).parent / "dataset.yaml"
        with open(path, "r") as f:
            info = yaml_load(f)
        return cls.from_dict(info)
