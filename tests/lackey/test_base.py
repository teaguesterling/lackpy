"""Tests for the Lackey base class."""

import pytest
from lackpy.lackey.base import Lackey
from lackpy.lackey.tool import Tool


class TestLackeyMetaclass:
    def test_collects_tools(self):
        class MyTask(Lackey):
            read = Tool()
            glob = Tool()
            def run(self): pass
        assert "read" in MyTask._tool_descriptors
        assert "glob" in MyTask._tool_descriptors

    def test_collects_params(self):
        class MyTask(Lackey):
            pattern: str = "**/*.py"
            verbose: bool = False
            def run(self): pass
        assert "pattern" in MyTask._param_specs
        assert MyTask._param_specs["pattern"]["type"] is str
        assert MyTask._param_specs["pattern"]["default"] == "**/*.py"

    def test_ignores_non_param_annotations(self):
        class MyTask(Lackey):
            read = Tool()
            returns: int
            def run(self) -> int: return 1
        assert "returns" not in MyTask._param_specs
        assert "read" not in MyTask._param_specs


class TestLackeyInstantiation:
    def test_default_params(self):
        class MyTask(Lackey):
            pattern: str = "**/*.py"
            def run(self): return self.pattern
        task = MyTask()
        assert task.pattern == "**/*.py"

    def test_override_params(self):
        class MyTask(Lackey):
            pattern: str = "**/*.py"
            def run(self): return self.pattern
        task = MyTask(pattern="src/*.py")
        assert task.pattern == "src/*.py"

    def test_unknown_param_raises(self):
        class MyTask(Lackey):
            def run(self): pass
        with pytest.raises(TypeError, match="Unknown parameter"):
            MyTask(nonexistent="value")


class TestLackeyToolResolution:
    def test_tools_resolved_with_callables(self):
        class MyTask(Lackey):
            read = Tool()
            def run(self):
                return self.read("test.py")
        def fake_read(path): return f"contents of {path}"
        task = MyTask()
        task._resolved_tools = {"read": fake_read}
        assert task.run() == "contents of test.py"


class TestLackeyInfo:
    def test_get_tool_names(self):
        class MyTask(Lackey):
            read = Tool()
            glob = Tool()
            def run(self): pass
        assert set(MyTask.get_tool_names()) == {"read", "glob"}

    def test_get_param_specs(self):
        class MyTask(Lackey):
            pattern: str = "**/*.py"
            max_depth: int = 10
            def run(self): pass
        specs = MyTask.get_param_specs()
        assert specs["pattern"]["type"] is str
        assert specs["max_depth"]["default"] == 10

    def test_get_returns(self):
        class MyTask(Lackey):
            returns: int
            def run(self) -> int: return 1
        assert MyTask.get_returns() is int
