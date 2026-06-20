"""Toolbox source precedence + drop-with-log collision merge (RFC 0002 §4, 3a)."""

from lackpy.tools.toolbox import ToolSpec, Toolbox


class _Src:
    """Minimal ToolSource: maps tool name -> callable."""

    def __init__(self, name: str, tools: dict):
        self._name = name
        self._tools = tools

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return True

    def discover(self):
        return [ToolSpec(name=n, provider=self._name) for n in self._tools]

    def resolve(self, spec):
        return self._tools[spec.name]


class _Prov:
    name = "p"

    def available(self) -> bool:
        return True

    def resolve(self, spec):
        return lambda: "registered"


def test_higher_precedence_wins_regardless_of_add_order():
    tb = Toolbox()
    tb.add_source(_Src("low", {"x": lambda: "low"}), precedence=1)
    tb.add_source(_Src("high", {"x": lambda: "high"}), precedence=10)
    assert tb.resolve("x")() == "high"

    tb2 = Toolbox()
    tb2.add_source(_Src("high", {"x": lambda: "high"}), precedence=10)
    tb2.add_source(_Src("low", {"x": lambda: "low"}), precedence=1)
    assert tb2.resolve("x")() == "high"


def test_equal_precedence_later_wins():
    tb = Toolbox()
    tb.add_source(_Src("a", {"x": lambda: "a"}), precedence=5)
    tb.add_source(_Src("b", {"x": lambda: "b"}), precedence=5)
    assert tb.resolve("x")() == "b"


def test_shadowed_tool_is_dropped_not_aliased():
    # v1: no qualified alias — the lower-precedence tool simply isn't reachable.
    tb = Toolbox()
    tb.add_source(_Src("high", {"x": lambda: "high"}), precedence=10)
    tb.add_source(_Src("low", {"x": lambda: "low"}), precedence=1)
    assert "low__x" not in tb.tools
    assert set(tb.tools) == {"x"}


def test_register_tool_outranks_even_high_precedence_source():
    tb = Toolbox()
    tb.register_provider(_Prov())
    tb.register_tool(ToolSpec(name="x", provider="p"))  # precedence inf
    tb.add_source(_Src("s", {"x": lambda: "src"}), precedence=999)
    assert tb.resolve("x")() == "registered"
