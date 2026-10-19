"""Tests for the unified service layer.

stepy100 Plan 2, Task 7: the kit is gone from the runtime. ``run_program`` runs
in the runtime namespace with the workspace ``PathPolicy`` and validates with
``allowed_names=set()``. A profile selects a model, its parameters and rules
only; a profile that names tools is an error.
"""

import inspect
import json
from pathlib import Path

import pytest

from lackpy.profiles import Profile
from lackpy.run.paths import load_policy
from lackpy.service import KIT_REMOVED, LITERATE_PENDING, STAGED_DSL_PENDING, LackpyService, ProfileError

REPO = Path(__file__).resolve().parents[1]

READ = "from pathlib import Path\nx = Path('test.txt').read_text()\nlen(x)"

_LITERATE = pytest.mark.xfail(
    reason="kit removed in stepy100 Plan 2; literate migration pending", strict=True)


@pytest.fixture
def tmp_home(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    return home


def _make_service(ws, home, calls=None):
    def loader(workspace):
        if calls is not None:
            calls.append(workspace)
        return load_policy(workspace, home=home, managed=None)
    return LackpyService(workspace=ws, path_policy_loader=loader)


@pytest.fixture
def service(tmp_path, tmp_home):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "test.txt").write_text("hello world")
    return _make_service(ws, tmp_home)


def _deny_read(ws, pattern="./test.txt"):
    (ws / ".claude").mkdir(exist_ok=True)
    (ws / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"deny": [f"Read({pattern})"]}}))


class TestValidate:
    def test_valid_program(self, service):
        result = service.validate(READ)
        assert result.valid, result.errors

    def test_kit_call_is_an_unknown_name(self, service):
        # Retired kit model: validate(..., profile=["read_file"]) allowed read_file.
        result = service.validate("x = read_file('test.txt')\nlen(x)")
        assert not result.valid

    def test_invalid_program(self, service):
        result = service.validate("import os")
        assert not result.valid


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_rules(self, service):
        # Retired kit model: asserted "read_file(" in the program.
        result = await service.generate("read file test.txt")
        assert "read_file" not in result.program
        assert service.validate(result.program).valid

    @pytest.mark.asyncio
    async def test_generate_no_match(self, service):
        with pytest.raises(RuntimeError):
            await service.generate("do something impossibly vague")


class TestRunProgram:
    @pytest.mark.asyncio
    async def test_run_valid_program(self, service):
        result = await service.run_program(READ)
        assert result.success, result.error
        assert result.output == 11
        assert [e.tool for e in result.trace.entries] == ["Path.read_text"]

    @pytest.mark.asyncio
    async def test_kit_call_fails_validation(self, service):
        result = await service.run_program("x = read_file('test.txt')\nlen(x)")
        assert not result.success
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_run_invalid_program(self, service):
        result = await service.run_program("import os")
        assert not result.success

    @pytest.mark.asyncio
    async def test_run_program_print_recovered_via_effective_output(self, service):
        # A program that prints its answer (instead of a bare last expression)
        # must not silently drop the value: the typed output is None, but the
        # printed text is captured and surfaced via effective_output.
        result = await service.run_program(
            "from pathlib import Path\nprint(Path('test.txt').read_text())")
        assert result.success, result.error
        assert result.output is None
        assert result.stdout == "hello world\n"
        assert result.effective_output == "hello world"

    @pytest.mark.asyncio
    async def test_relative_path_resolves_against_the_workspace_not_the_cwd(
            self, service, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = await service.run_program(READ)
        assert result.success, result.error
        assert result.output == 11

    @pytest.mark.asyncio
    async def test_path_policy_comes_from_the_workspace_settings(self, service):
        _deny_read(service._workspace)
        result = await service.run_program(READ)
        assert not result.success
        assert "not allowed in lackpy" in result.error

    @pytest.mark.asyncio
    async def test_path_policy_is_loaded_once_per_run(self, tmp_path, tmp_home):
        # Settings may change between runs, so each run loads them again.
        ws = tmp_path / "ws2"
        ws.mkdir()
        (ws / "test.txt").write_text("hello world")
        calls = []
        svc = _make_service(ws, tmp_home, calls)
        assert (await svc.run_program(READ)).success
        _deny_read(ws)
        assert not (await svc.run_program(READ)).success
        assert calls == [ws, ws]

    @pytest.mark.asyncio
    async def test_default_loader_is_load_policy_on_the_workspace(self, tmp_path, tmp_home, monkeypatch):
        import lackpy.service as service_module
        seen = []

        def fake(workspace, **kw):
            seen.append((workspace, kw))
            return load_policy(workspace, home=tmp_home, managed=None)

        monkeypatch.setattr(service_module, "load_policy", fake)
        svc = LackpyService(workspace=tmp_path)
        result = await svc.run_program("x = 1")
        assert result.success, result.error
        assert seen == [(tmp_path, {})]

    @pytest.mark.asyncio
    async def test_malformed_settings_file_is_a_failed_result_naming_the_file(self, service):
        settings = service._workspace / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text("{not json")
        result = await service.run_program("x = 1")
        assert not result.success
        assert str(settings) in result.error

    @pytest.mark.parametrize("settings", [
        '{"permissions": []}', '{"permissions": {"deny": "Read(x)"}}', '{"permissions": {"allow": [1]}}',
        '{"permissions": {"additionalDirectories": "x"}}', '[]', '"text"', '{"permissions": null}',
        '{"permissions": {"deny": [{"x": 1}]}}',
        '{"permissions": {"deny": ' + "[" * 100000 + "]" * 100000 + '}}',
    ], ids=["perms-list", "deny-str", "allow-int", "dirs-str", "top-list", "top-str", "perms-null",
            "deny-object", "deep-nesting"])
    @pytest.mark.asyncio
    async def test_wrong_typed_settings_are_a_failed_result_naming_the_file(self, service, settings):
        # Review M4: valid JSON with wrong types (and JSON nested past the recursion limit).
        path = service._workspace / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(settings)
        result = await service.run_program("x = 1")
        assert not result.success
        assert str(path) in result.error

    @pytest.mark.parametrize("settings", ['[paths]\ndeny = 3\n', 'paths = 3\n',
                                          '[paths]\ndeny = ' + "[" * 100000 + "]" * 100000 + '\n'],
                             ids=["deny-int", "paths-int", "deep-nesting"])
    @pytest.mark.asyncio
    async def test_wrong_typed_lackpy_config_paths_is_a_failed_result(self, tmp_path, tmp_home, settings):
        ws = tmp_path / "wsp"
        (ws / ".lackpy").mkdir(parents=True)
        svc = _make_service(ws, tmp_home)
        (ws / ".lackpy" / "config.toml").write_text(settings)  # after init: load_config ignores [paths]
        result = await svc.run_program("x = 1")
        assert not result.success
        assert str(ws / ".lackpy" / "config.toml") in result.error

    @pytest.mark.parametrize("exc", [TypeError("boom"), KeyError("k"), RecursionError("deep")])
    @pytest.mark.asyncio
    async def test_any_loader_exception_is_a_failed_result(self, tmp_path, exc):
        def loader(workspace):
            raise exc
        svc = LackpyService(workspace=tmp_path, path_policy_loader=loader)
        result = await svc.run_program("x = 1")
        assert not result.success
        assert "path policy" in result.error and type(exc).__name__ in result.error

    @pytest.mark.asyncio
    async def test_validation_always_runs(self, service):
        # Parked item 1: _execute_program skipped validation when allowed was None.
        rp = service._resolve_profile(None)
        result = await service._execute_program("import os\nos.getcwd()", rp, {})
        assert not result.success
        assert "Validation failed" in result.error

    @pytest.mark.asyncio
    async def test_non_data_output_is_returned_as_text(self, service):
        # Parked item 2: a Path stand-in output never reaches a host serializer.
        result = await service.run_program("from pathlib import Path\nPath('a')")
        assert result.success, result.error
        assert result.output == repr(Path("a"))
        json.dumps(result.output)


class TestDelegate:
    @pytest.mark.asyncio
    async def test_delegate_simple(self, service):
        # Retired kit model: asserted "read_file" in the program.
        result = await service.delegate("read file test.txt")
        assert result["success"]
        assert "read_file" not in result["program"]

    @pytest.mark.asyncio
    async def test_delegate_result_carries_stdout(self, service):
        # The delegate contract surfaces captured stdout so a printed answer is
        # never lost, even when the typed output is populated.
        result = await service.delegate("x", _program_override="print('hi')")
        assert result["success"], result["error"]
        assert result["stdout"] == "hi\n"

    @pytest.mark.asyncio
    async def test_delegate_with_params(self, service):
        # Task 5: a passed param needs the program's own params literal. This
        # program has none, so the run fails and names the key.
        result = await service.delegate("x", _program_override="y = 1", params={"prefix": "hello"})
        assert not result["success"]
        assert "'prefix'" in result["error"]

    @pytest.mark.asyncio
    async def test_delegate_with_params_teaches_and_runs_the_literal(self, service):
        # Task 8 (ruling b): the prompt teaches the params literal; the passed
        # value replaces the default the generator wrote.
        seen = {}

        class _Model:
            name = "fake"

            def available(self):
                return True

            async def generate(self, intent, namespace_desc, system_prompt_override=None, **kw):
                seen["system"] = system_prompt_override
                return "params = {'prefix': 'hello'}\nprint(params['prefix'])"

        service._inference_providers[:] = [_Model()]
        result = await service.delegate("print the prefix", params={"prefix": "bye"})
        assert result["success"], result["error"]
        assert result["stdout"] == "bye\n"
        assert "params = {'prefix': 'bye'}" in seen["system"]

    @pytest.mark.asyncio
    async def test_generate_does_not_admit_a_bare_param_name(self, service):
        # Task 8 (ruling b): params bind no bare names, so the generator's
        # validation must not allow them either.
        class _Model:
            name = "fake"

            def available(self):
                return True

            async def generate(self, intent, namespace_desc, **kw):
                return "prefix()"  # a call: only allowed names may be called

        service._inference_providers[:] = [_Model()]
        with pytest.raises(RuntimeError):
            await service.generate("print the prefix", params={"prefix": "bye"})

    def test_validate_takes_no_param_names(self, service):
        import inspect
        assert "param_names" not in inspect.signature(service.validate).parameters

    @pytest.mark.asyncio
    async def test_delegate_runs_a_path_program(self, service):
        result = await service.delegate("x", _program_override=READ)
        assert result["success"], result["error"]
        assert result["output"] == 11

    @pytest.mark.asyncio
    async def test_delegate_result_has_no_tool_grade(self, service):
        # The grade was the join of the kit's tool grades; no kit, no grade.
        result = await service.delegate("x", _program_override="y = 1")
        assert "grade" not in result

    @pytest.mark.asyncio
    async def test_delegate_validates_the_program(self, service):
        result = await service.delegate("x", _program_override="import os\nos.getcwd()")
        assert not result["success"]
        assert "Validation failed" in result["error"]


_TOOL_PROFILES = [
    ["read_file"],
    {"r": "read_file"},
    Profile(tools=["read_file"]),
    Profile(extra_tools=["read_file"]),
]


class TestProfiles:
    """A profile selects a model, its parameters and rules. Naming tools is an error."""

    @pytest.mark.parametrize("profile", _TOOL_PROFILES, ids=["list", "dict", "Profile.tools",
                                                             "Profile.extra_tools"])
    @pytest.mark.asyncio
    async def test_profile_naming_tools_is_an_error(self, service, profile):
        result = await service.run_program(READ, profile=profile)
        assert not result.success
        assert result.error == KIT_REMOVED

        verdict = service.validate(READ, profile=profile)
        assert not verdict.valid
        assert verdict.errors == [KIT_REMOVED]

        delegated = await service.delegate("x", profile=profile, _program_override=READ)
        assert not delegated["success"]
        assert delegated["error"] == KIT_REMOVED

        with pytest.raises(ProfileError, match="tool kits were removed"):
            await service.generate("read file test.txt", profile=profile)

        created = await service.create(READ, profile=profile, name="t")
        assert created == {"success": False, "errors": [KIT_REMOVED]}

    def test_ruled_message_text(self):
        assert KIT_REMOVED == ("tool kits were removed in lackpy stepy100; "
                               "use pathlib.Path (docs/concepts/paths.md)")

    @pytest.mark.parametrize("kit", sorted((REPO / ".lackpy" / "kits").glob("pluckit*.kit")),
                             ids=lambda p: p.stem)
    @pytest.mark.asyncio
    async def test_pluckit_kit_files_are_an_error(self, service, kit):
        # Parked item 8: the repo's pluckit kits stop working with the ruled message.
        kits = service._workspace / ".lackpy" / "kits"
        kits.mkdir(parents=True, exist_ok=True)
        (kits / kit.name).write_text(kit.read_text())
        result = await service.run_program("x = 1", profile=kit.stem)
        assert not result.success
        assert KIT_REMOVED in result.error

    @pytest.mark.asyncio
    async def test_configured_profile_with_tools_is_an_error(self, tmp_path, tmp_home):
        ws = tmp_path / "ws3"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profiles.old]\ntools = ["read_file"]\nmodel = "ollama/x"\n')
        svc = _make_service(ws, tmp_home)
        result = await svc.run_program("x = 1", profile="old")
        assert result.error == KIT_REMOVED

    @pytest.mark.asyncio
    async def test_configured_profile_selects_model_and_mode(self, tmp_path, tmp_home):
        ws = tmp_path / "ws4"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profiles.fast]\nmodel = "ollama/x"\nmode = "1-shot"\ntemperature = 0.1\n')
        (ws / "test.txt").write_text("hello world")
        svc = _make_service(ws, tmp_home)
        rp = svc._resolve_profile("fast")
        assert (rp.model, rp.mode, rp.temperature) == ("ollama/x", "1-shot", 0.1)
        assert rp.tools.tools == {} and rp.tools.callables == {}
        result = await svc.run_program(READ, profile="fast")
        assert result.success, result.error
        assert result.output == 11

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["nope", "none", "read_file"])
    async def test_unknown_profile_name_is_an_error(self, service, name):
        # Retired kit model: a bare name was a kit file, a tool, or "none" (empty kit).
        result = await service.run_program("x = 1", profile=name)
        assert not result.success
        assert repr(name) in result.error
        assert KIT_REMOVED in result.error

    def test_unconfigured_default_profile_means_no_profile(self, service):
        # The config default name ("debug") ships no [profiles.debug] table.
        rp = service._resolve_profile(None)
        assert rp.tools.tools == {} and rp.model is None

    @pytest.mark.parametrize("method", ["run_program", "delegate", "validate", "generate", "create"])
    def test_no_extra_tools_parameter(self, method):
        assert "extra_tools" not in inspect.signature(getattr(LackpyService, method)).parameters


class TestGetConfig:
    def test_returns_dict(self, service):
        config = service.get_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self, service):
        config = service.get_config()
        assert "inference_order" in config
        assert "profile_default" in config
        assert "sandbox_enabled" in config
        assert "config_dir" in config

    def test_config_dir_is_string(self, service):
        config = service.get_config()
        assert isinstance(config["config_dir"], str)


class TestProviderList:
    def test_returns_list(self, service):
        providers = service.provider_list()
        assert isinstance(providers, list)

    def test_providers_have_required_keys(self, service):
        providers = service.provider_list()
        # At minimum templates and rules are always present
        assert len(providers) >= 2
        for p in providers:
            assert "name" in p
            assert "plugin" in p
            assert "available" in p

    def test_templates_provider_present(self, service):
        providers = service.provider_list()
        names = [p["name"] for p in providers]
        assert "templates" in names

    def test_rules_provider_present(self, service):
        providers = service.provider_list()
        names = [p["name"] for p in providers]
        assert "rules" in names


class TestLanguageSpec:
    def test_returns_dict(self, service):
        spec = service.language_spec()
        assert isinstance(spec, dict)

    def test_has_spec_keys(self, service):
        spec = service.language_spec()
        assert "allowed_nodes" in spec
        assert "allowed_builtins" in spec


_LIT = Profile(execution="literate")


_OUTSIDE_MARK = "SECRET-OUTSIDE-THE-WORKSPACE"


def _escape_docs(outside):
    # Review C1, measured: on the literate axis a doc walked to object.__subclasses__(),
    # took the literate layer's read_file from a lambda's __globals__, and read a file
    # outside the workspace. No validator ran and no PathPolicy was loaded.
    return [
        "```lackpy\nn = len(().__class__.__base__.__subclasses__())\n```\n\nN={n}",
        f"```lackpy\ng = (lambda: 0).__globals__\nt = g['read_file']('{outside}')\n```\n\nT={{t}}",
    ]


class TestLiterateAxisRefused:
    """Review C1 / ruling: the literate axis is refused on every entry path until
    the literate interpreter migrates onto the runtime namespace and PathPolicy."""

    @pytest.fixture
    def outside(self, tmp_path):
        f = tmp_path / "outside.txt"
        f.write_text(_OUTSIDE_MARK)
        return f

    @pytest.mark.asyncio
    async def test_escape_fails_before_anything_runs(self, tmp_path, tmp_home, outside):
        ws = tmp_path / "wsl"
        ws.mkdir()
        calls = []
        svc = _make_service(ws, tmp_home, calls)
        for doc in _escape_docs(outside):
            result = await svc.run_program(doc, profile=_LIT)
            assert not result.success
            assert result.error == LITERATE_PENDING
            assert result.stdout == "" and result.output is None
            delegated = await svc.delegate("x", profile=_LIT, _program_override=doc)
            assert not delegated["success"]
            assert delegated["error"] == LITERATE_PENDING
            assert _OUTSIDE_MARK not in repr(delegated)
        assert calls == []  # refused before the policy load, so before any run

    @pytest.mark.asyncio
    async def test_configured_literate_profile_is_refused(self, tmp_path, tmp_home, outside):
        ws = tmp_path / "wsc"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profile]\ndefault = "lit"\n\n[profiles.lit]\nexecution = "literate"\n')
        svc = _make_service(ws, tmp_home)
        doc = _escape_docs(outside)[1]
        for profile in ("lit", None):  # by name, and as the configured default
            result = await svc.run_program(doc, profile=profile)
            assert result.error == LITERATE_PENDING
        assert svc.validate("x = 1", profile="lit").errors == [LITERATE_PENDING]
        with pytest.raises(ProfileError, match="literate"):
            await svc.generate("greet", profile="lit")
        assert (await svc.create("x = 1", profile="lit", name="t"))["errors"] == [LITERATE_PENDING]

    def test_message_names_the_pending_migration(self):
        assert "literate" in LITERATE_PENDING and "migrat" in LITERATE_PENDING

    def test_cli_refuses_a_configured_literate_profile(self, tmp_path, outside, capsys):
        from lackpy.cli import main
        ws = tmp_path / "wscli"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text('[profiles.lit]\nexecution = "literate"\n')
        prog = ws / "doc.py"
        prog.write_text(_escape_docs(outside)[1])
        code = main([str(prog), "--workspace", str(ws), "--profile", "lit"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert out["error"] == LITERATE_PENDING
        assert _OUTSIDE_MARK not in json.dumps(out)


class _SelectorProvider:
    """A model that answers every intent with an ast-select selector (review N2)."""
    name = "fake"

    def available(self):
        return True

    async def generate(self, intent, namespace_desc, **kw):
        return ".fn"


_DSL_SECRET = "SECRET-DSL"


class TestStagedDSLRefused:
    """Review N2 / ruling: mode "staged-dsl" ran AstSelectInterpreter over the process cwd
    at generation time with no PathPolicy, and the matched source became `program`.
    It is refused on every route until ast-select runs under the PathPolicy."""

    @pytest.fixture
    def ws(self, tmp_path, monkeypatch):
        ws = tmp_path / "wsd"
        (ws / ".claude").mkdir(parents=True)
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"deny": ["Read(denied.py)"]}}))
        (ws / "denied.py").write_text(f"def secret():\n    return '{_DSL_SECRET}'\n")
        monkeypatch.chdir(ws)  # ast-select read the process cwd
        ran = []
        from lackpy.interpreters.ast_select import AstSelectInterpreter
        original = AstSelectInterpreter.execute

        async def spy(self, *a, **k):
            ran.append(a)
            return await original(self, *a, **k)

        monkeypatch.setattr(AstSelectInterpreter, "execute", spy)
        monkeypatch.setattr(LackpyService, "_providers_for", lambda self, m, t: [_SelectorProvider()])
        import types
        return types.SimpleNamespace(path=ws, ran=ran)

    def _svc(self, ws, tmp_home, config=None):
        if config is not None:
            (ws.path / ".lackpy").mkdir(exist_ok=True)
            (ws.path / ".lackpy" / "config.toml").write_text(config)
        return _make_service(ws.path, tmp_home)

    def _assert_refused(self, result, ws):
        assert not result["success"]
        assert result["error"] == STAGED_DSL_PENDING
        assert _DSL_SECRET not in repr(result)
        assert ws.ran == []

    @pytest.mark.asyncio
    async def test_profile_mode_is_refused(self, ws, tmp_home):
        svc = self._svc(ws, tmp_home, '[profiles.d]\nmode = "staged-dsl"\n')
        self._assert_refused(await svc.delegate("x", profile="d"), ws)
        assert (await svc.run_program("x = 1", profile="d")).error == STAGED_DSL_PENDING

    @pytest.mark.asyncio
    async def test_explicit_mode_is_refused(self, ws, tmp_home):
        svc = self._svc(ws, tmp_home)
        self._assert_refused(await svc.delegate("x", mode="staged-dsl"), ws)

    @pytest.mark.asyncio
    async def test_inference_mode_is_refused(self, ws, tmp_home):
        svc = self._svc(ws, tmp_home, '[inference]\nmode = "staged-dsl"\n')
        self._assert_refused(await svc.delegate("x"), ws)

    @pytest.mark.asyncio
    async def test_generate_raises(self, ws, tmp_home):
        svc = self._svc(ws, tmp_home)
        with pytest.raises(ProfileError, match="staged-dsl"):
            await svc.generate("x", mode="staged-dsl")
        assert ws.ran == []

    def test_cli_mode_is_refused(self, ws, capsys):
        from lackpy.cli import main
        code = main(["-c", "find functions", "--mode", "staged-dsl", "--workspace", str(ws.path)])
        captured = capsys.readouterr()
        assert code == 1
        assert STAGED_DSL_PENDING in captured.out
        assert _DSL_SECRET not in captured.out + captured.err
        assert ws.ran == []

    def test_message_names_the_pending_migration(self):
        assert "staged-dsl" in STAGED_DSL_PENDING and "migrat" in STAGED_DSL_PENDING


class TestLiterateExecutionAxis:
    """Profile `execution == "literate"` routes run_program through the
    LiterateInterpreter, so the effect-ceiling gate (defaulting to the granted
    toolset's grade) and the write journal enforce end-to-end via the service.
    With no kit, the granted toolset is empty and the ceiling is w=0.
    """

    @_LITERATE
    @pytest.mark.asyncio
    async def test_write_doc_refused_under_a_literate_profile(self, service):
        # Retired kit model: the profile granted read_file (w=1). With no tools
        # the ceiling is w=0, so a @write doc (w=3) is refused before any cell runs.
        doc = "```lackpy @write(out.txt)\nhello\n```"
        result = await service.run_program(doc, profile=_LIT)
        assert not result.success
        assert "effect ceiling exceeded" in (result.error or "")
        assert not (service._workspace / "out.txt").exists()

    @_LITERATE
    @pytest.mark.asyncio
    async def test_within_ceiling_doc_executes_and_renders(self, service):
        # A pure doc (w=0) is within the ceiling -> it runs and renders.
        doc = "```lackpy\nx = 2 + 2\n```\n\nResult: {x}"
        result = await service.run_program(doc, profile=_LIT)
        assert result.success
        assert "Result: 4" in result.stdout

    @_LITERATE
    @pytest.mark.asyncio
    async def test_write_allowed_when_toolset_grade_permits(self, service):
        # Granting a w=3 tool raised the ceiling to w=3; a profile cannot grant tools now.
        doc = "```lackpy @write(out.txt)\nkept\n```"
        profile = Profile(tools=["read_file", "edit_file"], execution="literate")
        result = await service.run_program(doc, profile=profile)
        assert result.success, result.error
        assert (service._workspace / "out.txt").read_text() == "kept"

    @pytest.mark.asyncio
    async def test_one_shot_axis_unaffected(self, service):
        # The default execution axis still runs restricted Python through _execute.
        result = await service.run_program(READ)
        assert result.success

    @_LITERATE
    @pytest.mark.asyncio
    async def test_delegate_dispatches_literate_and_gate_refuses(self, service):
        # delegate routes through the same axis seam: a write doc under a literate
        # profile is refused by the gate (via _program_override, no LLM).
        result = await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy @write(o.txt)\nhi\n```")
        assert not result["success"]
        assert "effect ceiling exceeded" in (result["error"] or "")
        assert not (service._workspace / "o.txt").exists()

    @_LITERATE
    @pytest.mark.asyncio
    async def test_delegate_dispatches_literate_and_renders(self, service):
        result = await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy\nx = 2 + 2\n```\n\nResult: {x}")
        assert result["success"]
        assert "Result: 4" in result["stdout"]


class TestLiterateGenerationRouting:
    """generate() routes an execution=="literate" profile to the literate
    generation path (own hint + parse validation), not the Python dispatcher.
    """

    @_LITERATE
    @pytest.mark.asyncio
    async def test_literate_profile_uses_the_literate_generation_path(self, service, monkeypatch):
        from lackpy.infer.dispatch import GenerationResult
        seen = {}

        async def fake_gen_literate(intent, providers, resolved, params_desc, rules):
            seen["intent"] = intent
            return GenerationResult(program="Hello {x}", provider_name="fake", generation_time_ms=0.0)

        monkeypatch.setattr(service, "_generate_literate", fake_gen_literate)
        result = await service.generate("greet", profile=_LIT)
        assert seen.get("intent") == "greet"
        assert result.program == "Hello {x}"

    @pytest.mark.asyncio
    async def test_one_shot_profile_does_not_use_the_literate_path(self, service, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("_generate_literate must not run for a one-shot profile")

        monkeypatch.setattr(service, "_generate_literate", boom)
        # Retired kit model: asserted "read_file(" in the program.
        result = await service.generate("read file test.txt")
        assert "Path(" in result.program


class TestStripLiterateWrapper:
    def test_strips_leading_preamble(self):
        from lackpy.service import _strip_literate_wrapper
        out = _strip_literate_wrapper("Here is the document:\n```lackpy\nx=1\n```")
        assert out == "```lackpy\nx=1\n```"

    def test_strips_outer_markdown_wrapper(self):
        from lackpy.service import _strip_literate_wrapper
        assert _strip_literate_wrapper("```markdown\nHello {x}\n```") == "Hello {x}"

    def test_preserves_inner_lackpy_fences(self):
        from lackpy.service import _strip_literate_wrapper
        doc = "```lackpy @write(o.txt)\nv=1\n```"
        assert _strip_literate_wrapper(doc) == doc  # must NOT unwrap the code block

    def test_empty_is_empty(self):
        from lackpy.service import _strip_literate_wrapper
        assert _strip_literate_wrapper("   ") == ""


class _FakeKibitzer:
    """Minimal kibitzer stand-in: records the planned calls delegate sends to
    validate_calls, and returns a canned violation list."""

    def __init__(self, violations=None):
        self.mode = "test"
        self.seen_planned = None
        self._violations = violations or []

    def register_context(self, *a, **k):
        pass

    def validate_calls(self, planned):
        self.seen_planned = planned
        return self._violations

    def get_suggestions(self):
        return []

    def report_generation(self, *a, **k):
        pass

    def save(self):
        pass


class _Violation:
    def __init__(self, reason):
        self.reason = reason


class TestLiterateModePolicy:
    """delegate's kibitzer mode-policy pre-check applies to literate docs."""

    @_LITERATE
    def test_planned_calls_extracted_from_a_literate_doc(self, service):
        rp = service._resolve_profile(_LIT)
        calls = service._planned_tool_calls("```lackpy\ndata = read_file('t')\n```", rp)
        assert calls == ["read_file"]

    @_LITERATE
    def test_annotation_calls_are_extracted_too(self, service):
        # @write compiles to write_file(...) -- the compiled call is what policy sees.
        rp = service._resolve_profile(_LIT)
        calls = service._planned_tool_calls("```lackpy @write(o.txt)\nhi\n```", rp)
        assert "write_file" in calls

    @_LITERATE
    @pytest.mark.asyncio
    async def test_delegate_routes_literate_calls_to_mode_policy(self, service):
        fake = _FakeKibitzer()
        service._kibitzer = fake
        await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy\ndata = read_file('t')\n```")
        assert fake.seen_planned == [{"tool": "read_file", "input": {}}]

    @_LITERATE
    @pytest.mark.asyncio
    async def test_delegate_enforces_a_literate_mode_policy_violation(self, service):
        fake = _FakeKibitzer(violations=[_Violation("write_file blocked in test mode")])
        service._kibitzer = fake
        result = await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy @write(o.txt)\nhi\n```")
        assert not result["success"]
        assert "Kibitzer mode policy violation" in result["error"]
        assert "write_file blocked" in result["error"]

    @pytest.mark.asyncio
    async def test_one_shot_program_has_no_tool_calls_to_check(self, service):
        # Retired kit model: a one-shot read_file call reached validate_calls.
        # A valid program calls only builtins, imports and its own defs.
        fake = _FakeKibitzer()
        service._kibitzer = fake
        await service.delegate("read file", _program_override=READ)
        assert fake.seen_planned is None

    @_LITERATE
    def test_cooperative_limitation_indirect_calls_bypass(self, service):
        # Boundary pin: mode policy sees only DIRECT named calls. Attribute /
        # aliased / reflected invocations are NOT attributed to a tool, so they
        # slip the pre-check -- the same cooperative-planner limitation as the
        # effect gate. Soundness is nsjail + restricted-AST, not this pass.
        rp = service._resolve_profile(_LIT)
        attribute = service._planned_tool_calls("```lackpy\nimport os\nos.system('x')\n```", rp)
        aliased = service._planned_tool_calls("```lackpy\nw = write_file\nw('o', 'x')\n```", rp)
        assert "write_file" not in attribute      # os.system(...) surfaces nothing
        assert "write_file" not in aliased         # only the alias name 'w' surfaces


class TestResolveParams:
    """Review M7: only the single-key {"value": x} shape is unwrapped; dict params are data."""

    def test_single_key_value_shape_is_unwrapped(self, service):
        values, _ = service._resolve_params({"n": {"value": 3}})
        assert values == {"n": 3}

    def test_dict_param_with_a_value_key_is_kept(self, service):
        cfg = {"value": 1, "k": 2}
        values, _ = service._resolve_params({"cfg": cfg})
        assert values == {"cfg": cfg}
