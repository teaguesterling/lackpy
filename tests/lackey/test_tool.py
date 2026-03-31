"""Tests for the Tool descriptor."""

from lackpy.lackey.tool import Tool


class FakeProvider:
    pass


class TestToolSetName:
    def test_infers_name_from_attribute(self):
        class MyTask:
            read = Tool()
        assert MyTask.__dict__["read"]._name == "read"

    def test_infers_name_with_provider(self):
        class MyTask:
            find_defs = Tool(FakeProvider)
        assert MyTask.__dict__["find_defs"]._name == "find_defs"


class TestToolClassAccess:
    def test_class_access_returns_descriptor(self):
        class MyTask:
            read = Tool()
        assert isinstance(MyTask.read, Tool)


class TestToolProvider:
    def test_no_provider_means_builtin(self):
        t = Tool()
        assert t._provider is None

    def test_stores_provider_reference(self):
        t = Tool(FakeProvider)
        assert t._provider is FakeProvider


class TestToolRepr:
    def test_repr_no_provider(self):
        class MyTask:
            read = Tool()
        assert "read" in repr(MyTask.read)

    def test_repr_with_provider(self):
        class MyTask:
            find_defs = Tool(FakeProvider)
        assert "find_defs" in repr(MyTask.find_defs)
