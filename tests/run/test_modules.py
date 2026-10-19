"""Restricted ``__import__`` and module proxies (stepy100 Plan 2, Task 3).

A program never holds a real module. ``restricted_import`` serves a
``ModuleProxy`` for a module in ``dictionary.MODULES``, and refuses every other
import. The proxy exposes the dictionary members only, so an alias of the
module (``s = statistics``) reaches nothing that the static check would reject.
"""

import collections
import json
import types

import pytest

from lackpy.lang import dictionary, validate
from lackpy.run.modules import ModuleProxy, make_restricted_import
from lackpy.run.namespace import build_namespace
from lackpy.run.refusal import LackpyRefusal
from lackpy.run.runner import RestrictedRunner


@pytest.fixture
def imp():
    return make_restricted_import()


class TestRestrictedImport:
    @pytest.mark.parametrize("name", sorted(set(dictionary.MODULES) - {"pathlib"}))
    def test_dictionary_module_gives_a_proxy(self, imp, name):
        proxy = imp(name)
        assert isinstance(proxy, ModuleProxy)
        assert not isinstance(proxy, types.ModuleType)

    def test_same_proxy_within_a_run(self, imp):
        # Plain Python: `import json as a; import json as b; a is b`.
        assert imp("json") is imp("json")

    def test_os_refuses_with_the_dictionary_hint(self, imp):
        with pytest.raises(LackpyRefusal) as info:
            imp("os")
        assert info.value.hint == "use pathlib.Path"

    @pytest.mark.parametrize("name", ["sys", "subprocess", "collections.abc", "os.path", "builtins"])
    def test_other_modules_refuse(self, imp, name):
        with pytest.raises(LackpyRefusal):
            imp(name)

    def test_relative_import_refuses(self, imp):
        with pytest.raises(LackpyRefusal):
            imp("json", None, None, ("loads",), 1)

    @pytest.mark.parametrize("fromlist", [(), ("Path",)])
    def test_pathlib_refuses_until_the_path_policy(self, imp, fromlist):
        # Ruling C: no real pathlib module or Path class before Task 4.
        with pytest.raises(LackpyRefusal) as info:
            imp("pathlib", None, None, fromlist, 0)
        assert info.value.hint == "Path arrives with the PathPolicy"


class TestModuleProxy:
    @pytest.mark.parametrize("module, member", [
        ("statistics", "sys"), ("json", "codecs"), ("re", "functools"), ("collections", "abc"),
        ("json", "load"), ("json", "decoder"), ("math", "pi"),
    ])
    def test_non_member_refuses(self, imp, module, member):
        with pytest.raises(LackpyRefusal):
            getattr(imp(module), member)

    def test_member_is_the_real_member(self, imp):
        assert imp("json").loads is json.loads
        assert imp("json").loads("[1]") == [1]

    def test_missing_attribute_is_attribute_error(self, imp):
        # Plain Python gives AttributeError for a name the module lacks.
        with pytest.raises(AttributeError):
            imp("json").no_such_name

    @pytest.mark.parametrize("module", sorted(set(dictionary.MODULES) - {"pathlib"}))
    def test_no_member_is_a_module(self, imp, module):
        # A member that is a module would hand the program a real module.
        proxy = imp(module)
        for member in dictionary.MODULES[module]:
            try:
                value = getattr(proxy, member)
            except AttributeError:
                continue  # e.g. datetime.UTC before Python 3.11
            assert not isinstance(value, types.ModuleType), (module, member)

    def test_setattr_refuses(self, imp):
        proxy = imp("json")
        with pytest.raises(LackpyRefusal):
            proxy.loads = len
        with pytest.raises(LackpyRefusal):
            proxy.new_name = 1
        assert proxy.loads is json.loads

    def test_delattr_refuses(self, imp):
        with pytest.raises(LackpyRefusal):
            del imp("json").loads

    def test_version_is_the_real_value(self, imp):
        assert imp("json").__version__ == json.__version__

    def test_version_absent_is_attribute_error(self, imp):
        with pytest.raises(AttributeError):
            imp("math").__version__

    def test_repr_is_the_real_modules(self, imp):
        assert repr(imp("json")) == repr(json)


# Expressions over the collections classes, compared with plain Python (review M4, I1).
COLLECTIONS_EXPRESSIONS = [
    "repr(Counter())",
    "repr(Counter('abca'))",
    "repr(Counter('a') + Counter('b'))",
    "repr(Counter('aab') - Counter('a'))",
    "repr(Counter('ab') | Counter('a'))",
    "repr(Counter('ab') & Counter('a'))",
    "isinstance(Counter('a') + Counter('b'), Counter)",
    "isinstance(Counter('aab') - Counter('a'), Counter)",
    "isinstance(Counter('ab').copy(), Counter)",
    "repr(defaultdict(int))",
    "repr(defaultdict(Counter))",
    "repr(defaultdict(list, {'a': [1]}))",
    "isinstance(defaultdict(int) | {'a': 1}, defaultdict)",
    "repr(deque())",
    "repr(deque([1, 2, 3], maxlen=2))",
    "repr(deque([1]) + deque([2]))",
    "repr(OrderedDict())",
    "repr(OrderedDict(a=1))",
    "isinstance(OrderedDict(a=1) | {'b': 2}, OrderedDict)",
    "repr(Counter)",
    "repr(defaultdict)",
]


class TestCollectionsClasses:
    @pytest.mark.parametrize("name", ["Counter", "defaultdict", "deque", "OrderedDict"])
    def test_serves_the_real_class(self, imp, name):
        # Ruling (Review Focus 5 re-ruled): a subclass made Counter + Counter the
        # base class, a different result from plain Python.
        assert getattr(imp("collections"), name) is getattr(collections, name)

    @pytest.mark.parametrize("expression", COLLECTIONS_EXPRESSIONS)
    def test_matches_plain_python(self, expression):
        program = f"from collections import Counter, defaultdict, deque, OrderedDict\n{expression}"
        result = RestrictedRunner().run(program, namespace={})
        assert result.success, result.error
        plain = eval(expression, {  # noqa: S307
            "Counter": collections.Counter, "defaultdict": collections.defaultdict,
            "deque": collections.deque, "OrderedDict": collections.OrderedDict,
            "repr": repr, "isinstance": isinstance,
        })
        assert result.output == plain

    @pytest.mark.parametrize("program", [
        "from collections import Counter\nCounter.x = 1",
        "from collections import Counter\nC = Counter\nC.most_common = len",
        "import collections\ncollections.Counter.x = 1",
    ])
    def test_class_attribute_store_is_rejected(self, program):
        # The Review Focus 5 guard: the real class is shared by every run, so no
        # validated program may store an attribute on it.
        assert not validate(program).valid

    @pytest.mark.parametrize("name", ["setattr", "delattr", "getattr", "vars"])
    def test_no_reflection_builtin_can_store(self, name):
        assert name not in build_namespace(program_file=None, print_fn=print)["__builtins__"]


class TestNoRouteToTheRealModule:
    # Review M1: no attribute of a proxy may reach the real module.
    @pytest.mark.parametrize("name", [
        "_lackpy_module", "_lackpy_name", "_lackpy_members", "_module", "_members",
    ])
    def test_internal_names_do_not_give_the_module(self, imp, name):
        proxy = imp("json")
        try:
            value = getattr(proxy, name)
        except (AttributeError, LackpyRefusal):
            return
        pytest.fail(f"proxy.{name} gave {value!r}")

    @pytest.mark.parametrize("name", ["_lackpy_module", "_lackpy_name", "_lackpy_members"])
    def test_from_import_of_an_internal_name_fails(self, name):
        result = RestrictedRunner().run(f"from json import {name}\nrepr({name})", namespace={})
        assert not result.success
        assert "<module" not in str(result.output)

    def test_proxy_holds_no_instance_state(self, imp):
        proxy = imp("json")
        for slot in type(proxy).__slots__:
            if slot == "__weakref__":
                continue
            assert not hasattr(proxy, slot), slot


class TestNamespace:
    def test_builtins_bind_the_restricted_import(self):
        ns = build_namespace(program_file=None, print_fn=print)
        exec(compile("import json\nx = json.loads('[1]')", "<test>", "exec"), ns)  # noqa: S102
        assert ns["x"] == [1]
        assert isinstance(ns["json"], ModuleProxy)

    def test_each_namespace_has_its_own_proxies(self):
        a = build_namespace(program_file=None, print_fn=print)
        b = build_namespace(program_file=None, print_fn=print)
        exec("import collections", a)  # noqa: S102
        exec("import collections", b)  # noqa: S102
        assert a["collections"] is not b["collections"]
