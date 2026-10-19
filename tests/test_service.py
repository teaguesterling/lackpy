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
from lackpy.service import KIT_REMOVED, STAGED_DSL_PENDING, LackpyService, ProfileError

REPO = Path(__file__).resolve().parents[1]

READ = "from pathlib import Path\nx = Path('test.txt').read_text()\nlen(x)"

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


class TestLiterateAxisRuns:
    """Literate stepy Task 4 (D9): the literate axis runs through the service.

    It replaced ``TestLiterateAxisRefused`` (review C1 refused the axis with
    ``LITERATE_PENDING`` until the interpreter ran on the stepy100 runtime).
    The C1 escape document is gate 1 in ``tests/literate/test_gate.py``, on
    every entry point.
    """

    @pytest.mark.asyncio
    async def test_a_literate_run_loads_the_policy_through_the_service_seam(self, tmp_path, tmp_home):
        ws = tmp_path / "wsl"
        ws.mkdir()
        (ws / "in.txt").write_text("abc")
        calls = []
        svc = _make_service(ws, tmp_home, calls)
        doc = "```lackpy\nfrom pathlib import Path\nn = len(Path('in.txt').read_text())\n```\n\nN={n}"
        result = await svc.run_program(doc, profile=_LIT)
        assert result.success, result.error
        assert "N=3" in result.stdout
        assert calls == [ws]
        assert [e.tool for e in result.trace.entries] == ["Path.read_text"]

    @pytest.mark.asyncio
    async def test_a_policy_load_failure_is_a_failed_result(self, tmp_path):
        def boom(workspace):
            raise ValueError("bad settings")
        ws = tmp_path / "wsb"
        ws.mkdir()
        svc = LackpyService(workspace=ws, path_policy_loader=boom)
        result = await svc.run_program("```lackpy\nx = 1\n```", profile=_LIT)
        assert not result.success
        assert "bad settings" in result.error

    @pytest.mark.asyncio
    async def test_configured_literate_profile_runs(self, tmp_path, tmp_home):
        ws = tmp_path / "wsc"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profile]\ndefault = "lit"\n\n[profiles.lit]\nexecution = "literate"\n')
        svc = _make_service(ws, tmp_home)
        doc = "```lackpy\nx = 6 * 7\n```\n\nX={x}"
        for profile in ("lit", None):  # by name, and as the configured default
            result = await svc.run_program(doc, profile=profile)
            assert result.success, result.error
            assert "X=42" in result.stdout

    # Output (fix round 1, I1 and M1): run_program's last-expression rule. When the
    # run succeeded and the last code cell ends in an expression statement, the
    # output is that expression's value, data or its repr text; else None.

    @pytest.mark.asyncio
    @pytest.mark.parametrize("last, expected", [
        ("x", {"a": [1, 2]}),
        ("len(x['a'])", 2),
        ("x['a']", [1, 2]),
        ("y = 3", None),
    ])
    async def test_output_is_the_last_code_cells_trailing_expression(self, service, last, expected):
        doc = f"```lackpy\nx = {{'a': [1, 2]}}\n```\n\nText.\n\n```lackpy\n{last}\n```\n\nDone."
        result = await service.run_program(doc, profile=_LIT)
        assert result.success, result.error
        assert result.output == expected

    @pytest.mark.asyncio
    async def test_a_value_that_is_not_data_is_its_repr_text(self, service):
        doc = "```lackpy\nf = len\nf\n```"
        result = await service.run_program(doc, profile=_LIT)
        one_shot = await service.run_program("f = len\nf")
        assert result.success, result.error
        assert isinstance(result.output, str) and "len" in result.output
        assert result.output == one_shot.output

    @pytest.mark.asyncio
    async def test_the_trailing_expression_runs_once(self, service):
        doc = ("```lackpy\nfrom pathlib import Path\nprint('once')\n"
               "Path('o.txt').write_text('ab')\n```")
        result = await service.run_program(doc, profile=_LIT)
        assert result.success, result.error
        assert result.output == 2
        assert result.stdout.count("once") == 1
        assert [e.tool for e in result.trace.entries] == ["Path.write_text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("doc", [
        # Review I1, measured: each gave a value from a cell that never ran.
        "```lackpy\nx = 7\n```\n\n```lackpy\nfrom pathlib import Path\nPath('/etc/passwd').read_text()\n```\n\n```lackpy\nx\n```",
        "```lackpy\nx = 5\n```\n\n```lackpy\nnope + 1\nx\n```",
        "```lackpy\nx = 9\n```\n\n```lackpy\nz = 1 / 0\nx\n```",
    ])
    async def test_a_failed_run_has_no_output(self, service, doc):
        result = await service.run_program(doc, profile=_LIT)
        assert not result.success
        assert result.output is None
        delegated = await service.delegate("x", profile=_LIT, _program_override=doc)
        assert not delegated["success"]
        # delegate's effective output falls back to stdout (both axes); never
        # a value from a cell that did not run.
        assert delegated["output"] in (None, delegated["stdout"].strip())

    def test_pending_message_is_gone(self):
        import lackpy.profiles
        import lackpy.service
        assert not hasattr(lackpy.profiles, "LITERATE_PENDING")
        assert not hasattr(lackpy.service, "LITERATE_PENDING")


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


_RO = Profile(execution="literate", policy={"read_only": True})


class TestLiterateExecutionAxis:
    """Profile ``execution == "literate"`` routes run_program and delegate through
    the LiterateInterpreter on the stepy100 runtime: one CellSession under the
    workspace PathPolicy (D9). A literate cell can write what a one-shot program
    can; a read-only profile (``policy = {read_only = true}``) adds the
    ``Edit(//**)`` deny (D7, Ruling F).

    Retired kit model: the ceiling was the granted toolset's grade (a profile
    with read_file was w=1 and refused @write; one with edit_file was w=3).
    """

    @pytest.mark.asyncio
    async def test_write_doc_refused_under_a_read_only_profile(self, service):
        # Fix round 1, M5: the static ceiling refuses the write cell before it
        # runs (it is Unavailable); the document goes on.
        doc = "```lackpy @write(out.txt)\nhello\n```\n\n```lackpy\nafter = 1\n```\n\nA={after}"
        result = await service.run_program(doc, profile=_RO)
        assert not result.success
        assert "effect ceiling exceeded" in (result.error or "")
        assert "A=1" in result.stdout
        assert not (service._workspace / "out.txt").exists()

    def test_under_the_planner_the_read_only_policy_still_denies_writes(self, service):
        # The ceiling is a planner over the policy (ruling): the policy the
        # session gets for a read-only profile denies every write by itself.
        from lackpy.run.refusal import LackpyRefusal
        policy = service._load_path_policy(read_only=True)
        with pytest.raises(LackpyRefusal):
            policy.check(service._workspace / "out.txt", "write")
        policy.check(service._workspace / "test.txt", "read")

    @pytest.mark.asyncio
    async def test_a_read_only_profile_still_reads(self, service):
        doc = "```lackpy\nfrom pathlib import Path\nt = Path('test.txt').read_text()\n```\n\nT={t}"
        result = await service.run_program(doc, profile=_RO)
        assert result.success, result.error
        assert "T=hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_a_configured_read_only_profile_refuses_a_write(self, tmp_path, tmp_home):
        ws = tmp_path / "wsro"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profiles.ro]\nexecution = "literate"\npolicy = { read_only = true }\n')
        svc = _make_service(ws, tmp_home)
        doc = "```lackpy\nfrom pathlib import Path\nPath('o.txt').write_text('x')\n```"
        result = await svc.run_program(doc, profile="ro")
        assert not result.success
        assert not (ws / "o.txt").exists()

    @pytest.mark.asyncio
    async def test_read_only_applies_to_a_one_shot_profile_too(self, service):
        # Fail closed: the key is never ignored on the other axis.
        prog = "from pathlib import Path\nPath('o.txt').write_text('x')"
        result = await service.run_program(prog, profile=Profile(policy={"read_only": True}))
        assert not result.success
        assert not (service._workspace / "o.txt").exists()

    @pytest.mark.asyncio
    async def test_read_only_must_be_a_bool(self, service):
        result = await service.run_program(
            "```lackpy\nx = 1\n```", profile=Profile(execution="literate", policy={"read_only": "yes"}))
        assert not result.success
        assert "read_only" in result.error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("policy", [
        # Review I2, measured: the misspelled keys ran read-write; the
        # non-tables raised AttributeError out of run_program.
        {"readonly": True}, {"Read_Only": True}, {"read_only": True, "x": 1},
        "read_only", True, ["read_only"],
    ])
    async def test_a_malformed_policy_is_a_profile_error(self, service, policy):
        profile = Profile(execution="literate", policy=policy)
        doc = "```lackpy\nfrom pathlib import Path\nPath('k.txt').write_text('x')\n```"
        result = await service.run_program(doc, profile=profile)
        assert not result.success
        assert "policy" in result.error
        delegated = await service.delegate("x", profile=profile, _program_override=doc)
        assert not delegated["success"]
        assert not (service._workspace / "k.txt").exists()

    @pytest.mark.asyncio
    async def test_a_misspelled_configured_policy_is_a_profile_error(self, tmp_path, tmp_home):
        ws = tmp_path / "wsmis"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profiles.ro]\nexecution = "literate"\npolicy = { readonly = true }\n')
        svc = _make_service(ws, tmp_home)
        doc = "```lackpy\nfrom pathlib import Path\nPath('k.txt').write_text('x')\n```"
        result = await svc.run_program(doc, profile="ro")
        assert not result.success
        assert "readonly" in result.error
        assert not (ws / "k.txt").exists()

    def test_cli_with_a_non_table_policy_fails_cleanly(self, tmp_path, capsys, monkeypatch):
        from lackpy import service as service_module
        from lackpy.cli import main
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        monkeypatch.setattr(service_module, "load_policy",
                            lambda w: load_policy(w, home=home, managed=None))
        ws = tmp_path / "wscli"
        (ws / ".lackpy").mkdir(parents=True)
        (ws / ".lackpy" / "config.toml").write_text(
            '[profiles.ro]\nexecution = "literate"\npolicy = "read_only"\n')
        prog = tmp_path / "doc.md"
        prog.write_text("```lackpy\nx = 1\n```\n")
        code = main([str(prog), "--workspace", str(ws), "--profile", "ro"])
        out = json.loads(capsys.readouterr().out)
        assert code == 1
        assert "policy" in out["error"]

    @pytest.mark.asyncio
    async def test_within_ceiling_doc_executes_and_renders(self, service):
        doc = "```lackpy\nx = 2 + 2\n```\n\nResult: {x}"
        result = await service.run_program(doc, profile=_LIT)
        assert result.success
        assert "Result: 4" in result.stdout

    @pytest.mark.asyncio
    async def test_a_literate_profile_writes_under_the_policy(self, service):
        # Parity with a one-shot program: the policy, not a kit grade, decides.
        doc = "```lackpy @write(out.txt)\nkept\n```"
        result = await service.run_program(doc, profile=_LIT)
        assert result.success, result.error
        assert (service._workspace / "out.txt").read_text() == "kept"

    @pytest.mark.asyncio
    async def test_one_shot_axis_unaffected(self, service):
        # The default execution axis still runs restricted Python through _execute.
        result = await service.run_program(READ)
        assert result.success

    @pytest.mark.asyncio
    async def test_delegate_dispatches_literate_and_policy_refuses(self, service):
        result = await service.delegate(
            "x", profile=_RO, _program_override="```lackpy @write(o.txt)\nhi\n```")
        assert not result["success"]
        assert "effect ceiling exceeded" in (result["error"] or "")
        assert not (service._workspace / "o.txt").exists()

    @pytest.mark.asyncio
    async def test_delegate_dispatches_literate_and_renders(self, service):
        result = await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy\nx = 2 + 2\n```\n\nResult: {x}")
        assert result["success"]
        assert "Result: 4" in result["stdout"]

    @pytest.mark.asyncio
    async def test_delegate_reports_the_files_a_document_touched(self, service):
        doc = ("```lackpy\nfrom pathlib import Path\nt = Path('test.txt').read_text()\n"
               "Path('copy.txt').write_text(t)\n```")
        result = await service.delegate("x", profile=_LIT, _program_override=doc)
        assert result["success"], result["error"]
        assert result["files_read"] == ["test.txt"]
        assert result["files_modified"] == ["copy.txt"]


class TestLiterateGenerationRouting:
    """generate() routes an execution=="literate" profile to the literate
    generation path (own hint + parse validation), not the Python dispatcher.
    """

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

    def test_planned_calls_extracted_from_a_literate_doc(self, service):
        rp = service._resolve_profile(_LIT)
        calls = service._planned_tool_calls("```lackpy\ndata = read_file('t')\n```", rp)
        assert calls == ["read_file"]

    def test_annotation_calls_are_extracted_too(self, service):
        # Retired kit model: @write compiled to write_file(...), a named call.
        # Since literate stepy Task 3 it compiles to Path(...).write_text(...):
        # a method call, which names no planned call (see the limitation below).
        rp = service._resolve_profile(_LIT)
        calls = service._planned_tool_calls("```lackpy @write(o.txt)\nhi\n```", rp)
        assert calls == []

    @pytest.mark.asyncio
    async def test_delegate_routes_literate_calls_to_mode_policy(self, service):
        fake = _FakeKibitzer()
        service._kibitzer = fake
        await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy\ndata = read_file('t')\n```")
        assert fake.seen_planned == [{"tool": "read_file", "input": {}}]

    @pytest.mark.asyncio
    async def test_delegate_enforces_a_literate_mode_policy_violation(self, service):
        # Retired kit model: the doc was a @write (write_file); now a named call.
        fake = _FakeKibitzer(violations=[_Violation("read_file blocked in test mode")])
        service._kibitzer = fake
        result = await service.delegate(
            "x", profile=_LIT, _program_override="```lackpy\ndata = read_file('t')\n```")
        assert not result["success"]
        assert "Kibitzer mode policy violation" in result["error"]
        assert "read_file blocked" in result["error"]

    @pytest.mark.asyncio
    async def test_one_shot_program_has_no_tool_calls_to_check(self, service):
        # Retired kit model: a one-shot read_file call reached validate_calls.
        # A valid program calls only builtins, imports and its own defs.
        fake = _FakeKibitzer()
        service._kibitzer = fake
        await service.delegate("read file", _program_override=READ)
        assert fake.seen_planned is None

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


class TestFinalReviewFixes:
    """Fixes from the Plan 2 whole-branch review (final-review.md)."""

    @pytest.mark.asyncio
    async def test_validation_error_carries_the_hint(self, service):
        # I3, spec D17: the static message names the approved alternative.
        from lackpy.lang import validate
        program = 'open("a.txt").read()'
        (violation,) = [v for v in validate(program, allowed_names=set()).violations if v.hint]
        result = await service.run_program(program)
        assert not result.success
        assert f"{violation.message} ({violation.hint})" in result.error
        assert "Path(p).read_text()" in result.error

    @pytest.mark.asyncio
    async def test_a_symlink_loop_in_a_deny_rule_is_a_failed_result(self, tmp_path, tmp_home):
        # I4: the reviewer's reproduction. The refusal escaped run_program.
        ws = tmp_path / "loopws"
        ws.mkdir()
        (ws / "loop").symlink_to(ws / "loop")
        (ws / ".claude").mkdir()
        (ws / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"deny": ["Read(./loop/secret)"]}}))
        result = await _make_service(ws, tmp_home).run_program("x = 1\nx")
        assert not result.success
        assert "Read(./loop/secret)" in result.error

    @pytest.mark.asyncio
    async def test_a_refusal_from_the_loader_is_a_failed_result(self, tmp_path):
        from lackpy.run.refusal import LackpyRefusal

        def loader(workspace):
            raise LackpyRefusal("access to loop")
        svc = LackpyService(workspace=tmp_path, path_policy_loader=loader)
        result = await svc.run_program("x = 1")
        assert not result.success
        assert "access to loop" in result.error

    def test_rule_names_resolve_in_validate(self, service):
        # M1: MCP advertises rules as list[str]; a string raised TypeError.
        assert not service.validate("for i in []:\n    pass", rules=["no_loops"]).valid
        assert service.validate("x = 1", rules=["no_loops"]).valid
        assert not service.validate("len([])\nlen([])", rules=["max_calls(1)"]).valid

    def test_unknown_rule_name_is_an_invalid_result(self, service):
        result = service.validate("x = 1", rules=["no_such_rule"])
        assert not result.valid
        assert "no_such_rule" in result.errors[0]

    @pytest.mark.asyncio
    async def test_unknown_rule_name_is_a_failed_run(self, service):
        result = await service.run_program("x = 1", rules=["no_such_rule"])
        assert not result.success
        assert "no_such_rule" in result.error

    @pytest.mark.asyncio
    async def test_unknown_rule_name_is_a_failed_delegate(self, service):
        result = await service.delegate("x", rules=["no_such_rule"], _program_override="x = 1")
        assert result["success"] is False
        assert "no_such_rule" in result["error"]

    @pytest.mark.asyncio
    async def test_delegate_reports_files_read_and_modified(self, service):
        # M3: the Path stand-in's trace fills files_read and files_modified.
        program = ("from pathlib import Path\n"
                   "t = Path('test.txt').read_text()\n"
                   "Path('out.txt').write_text(t)\n"
                   "Path('test.txt').read_bytes()\n"
                   "Path('missing.txt').exists()\n"
                   "len(t)")
        result = await service.delegate("copy", _program_override=program)
        assert result["success"], result["error"]
        assert result["files_read"] == ["test.txt"]
        assert result["files_modified"] == ["out.txt"]

    @pytest.mark.asyncio
    async def test_kibitzer_violation_result_has_the_common_keys(self, service, monkeypatch):
        # M4: every other delegate result has stdout, output and trace.
        service._kibitzer = _FakeKibitzer(violations=[_Violation("blocked")])
        monkeypatch.setattr(service, "_planned_tool_calls", lambda program, rp: ["x"])
        result = await service.delegate("x", _program_override="x = 1")
        assert result["success"] is False
        assert {"stdout", "output", "trace"} <= result.keys()
        assert (result["stdout"], result["output"], result["trace"]) == ("", None, [])

    @pytest.mark.asyncio
    async def test_delegate_reapplies_the_callers_rules_at_execution(self, service):
        # M5: a rule applied at generation is applied at the execution boundary too.
        def reject_all(tree):
            return ["rejected by the caller's rule"]
        result = await service.delegate("x", rules=[reject_all], _program_override="x = 1")
        assert result["success"] is False
        assert "rejected by the caller's rule" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["../../../x", "a/b", "a\\b", "..", "/abs", "nul\x00"])
    async def test_create_refuses_a_name_that_is_not_a_plain_file_name(self, service, tmp_path, name):
        # M7: the name was joined unsanitized, so ../ escaped .lackpy/templates.
        before = {p for p in tmp_path.rglob("*")}
        result = await service.create("x = 1", name=name)
        assert result["success"] is False
        assert "name" in result["errors"][0]
        assert {p for p in tmp_path.rglob("*")} == before

    @pytest.mark.asyncio
    async def test_create_keeps_a_plain_name(self, service):
        result = await service.create("x = 1", name="count_lines")
        assert result["success"], result
        assert Path(result["path"]).name == "count_lines.tmpl"
