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

from lackpy.lang import dictionary
from lackpy.run.modules import ModuleProxy, make_restricted_import
from lackpy.run.namespace import build_namespace
from lackpy.run.refusal import LackpyRefusal


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


class TestCollectionsClasses:
    @pytest.mark.parametrize("name", ["Counter", "defaultdict", "deque", "OrderedDict"])
    def test_fresh_subclass_per_run(self, name):
        base = getattr(collections, name)
        first = getattr(make_restricted_import()("collections"), name)
        second = getattr(make_restricted_import()("collections"), name)
        assert first is not base and first is not second
        assert issubclass(first, base)
        assert first.__name__ == base.__name__
        assert first.__qualname__ == base.__qualname__

    def test_reprs_match_plain_python(self, imp):
        c = imp("collections")
        assert repr(c.Counter("ab")) == repr(collections.Counter("ab"))
        assert repr(c.deque([1])) == repr(collections.deque([1]))
        assert repr(c.defaultdict(int)) == repr(collections.defaultdict(int))
        assert repr(c.OrderedDict(a=1)) == repr(collections.OrderedDict(a=1))

    def test_class_state_does_not_cross_runs(self):
        first = make_restricted_import()("collections").Counter
        first.marker = 1
        second = make_restricted_import()("collections").Counter
        assert not hasattr(second, "marker")
        assert not hasattr(collections.Counter, "marker")


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
        assert a["collections"].Counter is not b["collections"].Counter
