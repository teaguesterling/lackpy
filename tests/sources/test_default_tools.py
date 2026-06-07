"""Shipped default tools: load as data, populate the service, override, no hardcode."""

import lackpy.service as service_module
from lackpy.config import LackpyConfig
from lackpy.service import LackpyService
from lackpy.sources import load_default_tool_defs


def test_no_hardcoded_builtin_tools_list():
    # The whole point of the slice: tools come from sources, not a Python list.
    assert not hasattr(service_module, "_BUILTIN_TOOLS")


def test_default_defs_load_from_packaged_toml():
    names = {d["name"] for d in load_default_tool_defs()}
    assert {"read_file", "find_files", "write_file", "edit_file"} <= names


def test_service_has_builtins_by_default(tmp_path):
    svc = LackpyService(workspace=tmp_path)
    names = {t["name"] for t in svc.toolbox_list()}
    assert {"read_file", "find_files", "write_file", "edit_file"} <= names
    # builtins are sourced as data from lackpy.sources.builtins
    assert svc.toolbox.resolve("read_file").__module__ == "lackpy.sources.builtins"


async def test_default_read_file_runs_end_to_end(tmp_path):
    (tmp_path / "hello.txt").write_text("world")
    svc = LackpyService(workspace=tmp_path)
    res = await svc.run_program("content = read_file('hello.txt')\ncontent", kit=["read_file"])
    assert res.success
    assert res.output == "world"


def test_load_config_parses_user_tools_and_service_picks_them_up(tmp_path):
    # Exercises config.load_config's [[tools]] parse (not bypassed via LackpyConfig)
    # and the service auto-loading it from a real .lackpy/config.toml.
    from lackpy.config import load_config

    cfgdir = tmp_path / ".lackpy"
    cfgdir.mkdir()
    (cfgdir / "config.toml").write_text(
        '[[tools]]\n'
        'name = "list_glob"\n'
        'provider = "python"\n'
        'module = "lackpy.sources.builtins"\n'
        'function = "find_files"\n'
        'returns = "list[str]"\n'
        'grade_w = 1\n'
        'effects_ceiling = 1\n'
        'args = [{ name = "pattern", type = "str" }]\n'
    )
    cfg = load_config(tmp_path)
    assert any(t["name"] == "list_glob" for t in cfg.tools)

    svc = LackpyService(workspace=tmp_path)
    names = {t["name"] for t in svc.toolbox_list()}
    assert "list_glob" in names
    assert svc.toolbox.resolve("list_glob").__name__ == "find_files"


def test_user_tool_overrides_default(tmp_path):
    # A user [[tools]] entry with a default's name wins (source order: default then config).
    cfg = LackpyConfig(
        tools=[
            {
                "name": "read_file",
                "provider": "python",
                "module": "lackpy.sources.builtins",
                "function": "find_files",
                "returns": "list[str]",
                "grade_w": 1,
                "effects_ceiling": 1,
                "args": [{"name": "path", "type": "str"}],
            }
        ]
    )
    svc = LackpyService(workspace=tmp_path, config=cfg)
    assert svc.toolbox.resolve("read_file").__name__ == "find_files"
