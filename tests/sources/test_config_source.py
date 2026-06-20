"""Config-defined tool source: discovery, resolution, grade, override, guard."""

import pytest

from lackpy.tools.toolbox import Toolbox
from lackpy.sources.config import ConfigToolSource

DEFS = [
    {
        "name": "myread",
        "provider": "python",
        "module": "lackpy.sources.builtins",
        "function": "read_file",
        "description": "read a file",
        "returns": "str",
        "grade_w": 1,
        "effects_ceiling": 1,
        "args": [{"name": "path", "type": "str", "description": "path"}],
    }
]


def test_discover_builds_full_spec():
    spec = ConfigToolSource(DEFS).discover()[0]
    assert spec.name == "myread"
    assert spec.grade_w == 1 and spec.effects_ceiling == 1
    assert [a.name for a in spec.args] == ["path"]
    assert spec.provider_config == {
        "module": "lackpy.sources.builtins",
        "function": "read_file",
    }


def test_available_reflects_defs():
    assert ConfigToolSource([]).available() is False
    assert ConfigToolSource(DEFS).available() is True


def test_grade_defaults_conservative():
    spec = ConfigToolSource(
        [{"name": "x", "module": "lackpy.sources.builtins", "function": "read_file"}]
    ).discover()[0]
    assert spec.grade_w == 3 and spec.effects_ceiling == 3


def test_missing_name_raises():
    with pytest.raises(ValueError):
        ConfigToolSource([{"module": "m", "function": "f"}]).discover()


def test_resolve_returns_working_callable(tmp_path):
    src = ConfigToolSource(DEFS)
    fn = src.resolve(src.discover()[0])
    p = tmp_path / "f.txt"
    p.write_text("hi")
    assert fn(str(p)) == "hi"


def test_toolbox_add_source_then_resolve(tmp_path):
    tb = Toolbox()
    tb.add_source(ConfigToolSource(DEFS))
    assert "myread" in tb.tools
    p = tmp_path / "f.txt"
    p.write_text("yo")
    assert tb.resolve("myread")(str(p)) == "yo"


def test_later_source_overrides_earlier():
    tb = Toolbox()
    tb.add_source(
        ConfigToolSource(
            [{"name": "t", "module": "lackpy.sources.builtins", "function": "read_file"}]
        )
    )
    tb.add_source(
        ConfigToolSource(
            [{"name": "t", "module": "lackpy.sources.builtins", "function": "find_files"}]
        )
    )
    assert tb.resolve("t").__name__ == "find_files"
