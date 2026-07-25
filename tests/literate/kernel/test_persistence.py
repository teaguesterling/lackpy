"""Option B Stage 1 acceptance — kernel persistence-readiness.

Three pinned contracts (design: pages/private/design-optionB-kernel-aidr.md):

1. SERIALIZATION: Hole / ErrorValue / Unavailable round-trip EXACTLY through
   to_dict/from_dict (all-string fields + a ``__kind__`` discriminator).
   BindingVersion serializes with the DECISION-1 best-effort value strategy:
   forgiveness values exactly, JSON-able values exactly, anything else as an
   inspectable ``__nonserializable__`` repr marker — never raising.
2. IDENTITY: session_id/document_id thread from LiterateSession into the
   ledger AND the binding history; every asserted BindingVersion is stamped
   with a ``created_at`` (same float-epoch time source as the ledger).
3. BACKEND PROTOCOL: a PersistenceBackend receives every serialized ledger
   row + binding version at the two choke points (record / assert_binding),
   strictly AFTER the in-memory operation; with NO backend the kernel is
   byte-identical to the un-wired path (the no-op-when-unplugged guarantee).

The dict-backend test at the bottom is the Stage-1 acceptance: it proves
persistence-readiness against a trivial in-memory backend, with zero
dependency on any other project.
"""

import json
from dataclasses import replace

import pytest

from lackpy.interpreters.base import ExecutionContext
from lackpy.interpreters.literate import LiterateSession
from lackpy.interpreters.literate.kernel import (
    BindingVersion,
    BindingVersions,
    ErrorValue,
    Hole,
    Ledger,
    PersistenceBackend,
    PersistentBindingVersions,
    PersistentLedger,
    Unavailable,
    deserialize_value,
    ledger_entry_to_dict,
    serialize_value,
)
from lackpy.interpreters.literate.kernel.forgiveness import forgiveness_from_dict


@pytest.fixture
def context(tmp_path):
    return ExecutionContext(base_dir=tmp_path)


class DictBackend:
    """The simplest possible PersistenceBackend: lists of received dicts.

    Structurally satisfies the protocol without importing/inheriting anything
    from the kernel — exactly the position a third-party plugin is in.
    """

    def __init__(self):
        self.entries = []
        self.bindings = []

    def persist_ledger_entry(self, entry):
        self.entries.append(entry)

    def persist_binding(self, binding):
        self.bindings.append(binding)

    def binding_table(self):
        """Upsert view keyed on (name, version) — last write wins, as the
        protocol specifies for supersession updates."""
        table = {}
        for b in self.bindings:
            table[(b["name"], b["version"])] = b
        return table


class TestForgivenessRoundTrip:
    """Contract 1a: forgiveness kinds round-trip EXACTLY."""

    @pytest.mark.parametrize(
        "value",
        [
            Hole("x"),
            Hole("y", blocked_by=("a", "b")),
            ErrorValue("z", "ZeroDivisionError: division by zero"),
            ErrorValue("w", "SyntaxError: bad", error_phase="compile"),
            Unavailable("q", "effect ceiling exceeded: net"),
        ],
        ids=["hole", "chained-hole", "error", "error-phase", "unavailable"],
    )
    def test_round_trip_exact(self, value):
        data = value.to_dict()
        json.dumps(data)  # the serialized form must be JSON-safe
        restored = forgiveness_from_dict(data)
        assert restored == value
        assert type(restored) is type(value)

    def test_discriminators(self):
        assert Hole("x").to_dict()["__kind__"] == "hole"
        assert ErrorValue("x", "E: m").to_dict()["__kind__"] == "error"
        assert Unavailable("x", "r").to_dict()["__kind__"] == "unavailable"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            forgiveness_from_dict({"__kind__": "gremlin"})


class TestSerializeValue:
    """Contract 1b: DECISION 1 (option a) — best-effort value_json."""

    @pytest.mark.parametrize(
        "value",
        [42, 3.5, "text", None, True, [1, 2, 3], {"a": [1, {"b": None}]}],
    )
    def test_jsonable_round_trips_exactly(self, value):
        stored = serialize_value(value)
        json.dumps(stored)
        assert deserialize_value(stored) == value

    def test_forgiveness_round_trips_exactly(self):
        hole = Hole("x", blocked_by=("up",))
        stored = serialize_value(hole)
        assert stored["__kind__"] == "hole"
        assert deserialize_value(stored) == hole

    @pytest.mark.parametrize(
        "value",
        [lambda: 1, object(), {1, 2, 3}],
        ids=["lambda", "object", "set"],
    )
    def test_nonserializable_becomes_marker_never_raises(self, value):
        stored = serialize_value(value)  # must not raise
        assert stored["__nonserializable__"] is True
        assert stored["__repr__"] == repr(value)
        assert stored["type"] == type(value).__name__
        json.dumps(stored)
        # Deserialization keeps the marker — inspectable, NOT reconstructed.
        assert deserialize_value(stored) == stored

    def test_open_file_becomes_marker(self, tmp_path):
        with open(tmp_path / "f.txt", "w") as handle:
            stored = serialize_value(handle)
        assert stored["__nonserializable__"] is True
        assert stored["type"] == "TextIOWrapper"

    def test_hostile_repr_still_never_raises(self):
        class Hostile:
            def __repr__(self):
                raise RuntimeError("no repr for you")

        stored = serialize_value(Hostile())
        assert stored["__nonserializable__"] is True
        assert stored["type"] == "Hostile"

    def test_tuple_normalizes_to_json_form(self):
        # Documented normalization: what is stored is what survives JSON.
        assert serialize_value((1, 2)) == [1, 2]


class TestBindingVersionSerialization:
    """Contract 1c: the serialized BindingVersion carries the AIDR columns."""

    def test_to_dict_carries_aidr_columns_and_identity(self):
        versions = BindingVersions(
            session_id="S1", document_id="D1", clock=lambda: 7.0
        )
        new, _ = versions.assert_binding("x", 41)
        data = versions.serialize(new)
        assert data == {
            "name": "x",
            "version": 1,
            "kind": "value",
            "value_json": 41,
            "superseded_by": None,
            "created_at": 7.0,
            "session_id": "S1",
            "document_id": "D1",
        }
        json.dumps(data)

    @pytest.mark.parametrize(
        "value,kind",
        [
            (41, "value"),
            (Hole("x"), "hole"),
            (ErrorValue("x", "E: m"), "error"),
            (Unavailable("x", "r"), "unavailable"),
        ],
    )
    def test_kind_accessor_and_round_trip(self, value, kind):
        versions = BindingVersions(clock=lambda: 1.5)
        new, _ = versions.assert_binding("x", value)
        assert new.kind == kind
        data = versions.serialize(new)
        assert data["kind"] == kind
        assert BindingVersion.from_dict(data) == new  # exact round-trip

    def test_superseded_is_lifecycle_not_kind(self):
        versions = BindingVersions(clock=lambda: 1.0)
        versions.assert_binding("x", 1)
        _, prior = versions.assert_binding("x", 2)
        assert prior.superseded and prior.superseded_by == 2
        assert prior.kind == "value"  # kind is the VALUE kind, not lifecycle
        assert versions.serialize(prior)["superseded_by"] == 2

    def test_nonserializable_value_round_trips_to_marker(self):
        versions = BindingVersions(clock=lambda: 1.0)
        new, _ = versions.assert_binding("f", lambda: 1)
        data = versions.serialize(new)
        assert data["value_json"]["__nonserializable__"] is True
        restored = BindingVersion.from_dict(data)
        # The marker (not the lambda) comes back — expected and honest.
        assert restored.value == data["value_json"]
        assert (restored.name, restored.version, restored.created_at) == (
            "f", 1, 1.0,
        )

    def test_created_at_uses_wall_clock_by_default(self):
        import time

        before = time.time()
        new, _ = BindingVersions().assert_binding("x", 1)
        assert before <= new.created_at <= time.time()


class TestIdentityThreading:
    """Contract 2: session_id/document_id flow from LiterateSession."""

    @pytest.mark.asyncio
    async def test_session_identity_reaches_ledger_and_versions(self, context):
        session = LiterateSession(context, session_id="S1", document_id="D1")
        await session.step("```lackpy\nx = 1\n```")

        entries = session.ledger.entries()
        assert entries  # selftest summary + round entries
        assert all(e.session_id == "S1" for e in entries)
        assert all(e.document_id == "D1" for e in entries)

        assert session.versions.session_id == "S1"
        assert session.versions.document_id == "D1"
        (v1,) = session.versions.history("x")
        data = session.versions.serialize(v1)
        assert data["session_id"] == "S1"
        assert data["document_id"] == "D1"
        assert data["created_at"] > 0

    @pytest.mark.asyncio
    async def test_default_identity_unchanged_and_shared(self, context):
        session = LiterateSession(context)  # no identity given — today's path
        await session.step("```lackpy\nx = 1\n```")
        generated = session.ledger.session_id
        assert generated  # auto-generated, as before
        assert session.ledger.document_id is None
        # The binding history shares the ledger's generated identity.
        assert session.versions.session_id == generated
        assert session.versions.document_id is None


DOC_ALL_KINDS = (
    "```lackpy\nx = 1\n```\n\n"      # real value
    "```lackpy\nx = 2\n```\n\n"      # re-assertion → supersession
    "```lackpy\ny = missing\n```\n\n"  # origin hole + chained hole
    "```lackpy\nz = 1 / 0\n```"      # runtime error → ErrorValue
)


class TestNoOpWhenUnplugged:
    """Contract 3a: NO backend → kernel byte-identical to the un-wired path."""

    @pytest.mark.asyncio
    async def test_backend_wiring_changes_nothing_observable(self, context, tmp_path):
        (tmp_path / "wired").mkdir(exist_ok=True)
        plain = LiterateSession(context, session_id="S")
        wired = LiterateSession(
            ExecutionContext(base_dir=tmp_path / "wired"),
            session_id="S",
            persistence=DictBackend(),
        )
        r_plain = await plain.step(DOC_ALL_KINDS)
        r_wired = await wired.step(DOC_ALL_KINDS)

        assert r_plain == r_wired  # the full StepResult, byte for byte
        assert plain.rendered == wired.rendered
        # Ledger rows identical modulo wall-clock timestamps.
        strip = lambda e: replace(e, created_at=0.0)  # noqa: E731
        assert [strip(e) for e in plain.ledger.entries()] == [
            strip(e) for e in wired.ledger.entries()
        ]
        # Version histories identical modulo wall-clock timestamps.
        assert plain.versions.names() == wired.versions.names()
        strip_v = lambda v: replace(v, created_at=0.0)  # noqa: E731
        for name in plain.versions.names():
            assert [strip_v(v) for v in plain.versions.history(name)] == [
                strip_v(v) for v in wired.versions.history(name)
            ]


class TestDictBackendAcceptance:
    """Contract 3b: THE Stage-1 acceptance — a trivial dict backend receives
    everything, matching the in-memory stores, with zero external deps."""

    def test_dict_backend_satisfies_protocol(self):
        assert isinstance(DictBackend(), PersistenceBackend)

    def test_persistent_classes_call_backend_after_the_fact(self):
        backend = DictBackend()
        ledger = PersistentLedger(backend, session_id="S", clock=lambda: 2.0)
        entry = ledger.record("executed", name="cell", detail={"i": 0})
        assert ledger.entries() == [entry]  # in-memory behavior unchanged
        assert backend.entries == [ledger_entry_to_dict(entry)]
        assert backend.entries[0]["session_id"] == "S"

        versions = PersistentBindingVersions(
            backend, session_id="S", clock=lambda: 3.0
        )
        versions.assert_binding("x", 1)
        versions.assert_binding("x", 2)
        # Supersession sends the updated prior THEN the new current version.
        assert [(b["version"], b["superseded_by"]) for b in backend.bindings] == [
            (1, None), (1, 2), (2, None),
        ]

    @pytest.mark.asyncio
    async def test_full_session_streams_to_backend(self, context):
        backend = DictBackend()
        session = LiterateSession(
            context, session_id="S1", document_id="D1", persistence=backend
        )
        result = await session.step(DOC_ALL_KINDS)
        assert not result.ok  # holes + error reified → aggregate Left

        # (1) Every ledger row reached the backend, in order, serialized.
        assert backend.entries == [
            ledger_entry_to_dict(e) for e in session.ledger.entries()
        ]
        recorded_types = {e["entry_type"] for e in backend.entries}
        assert {"executed", "hole_opened", "error_reified", "superseded"} <= (
            recorded_types
        )

        # (2) The backend's upsert view matches the in-memory history exactly.
        table = session.versions
        persisted = backend.binding_table()
        assert table.names()  # x, y, z, missing (+ any kernel-asserted names)
        for name in table.names():
            for version in table.history(name):
                key = (name, version.version)
                assert persisted[key] == table.serialize(version)
                # (3) Round-trip: the persisted dict reconstructs the version.
                assert BindingVersion.from_dict(persisted[key]) == version

        # (4) The interesting kinds all made it through, marked correctly.
        kinds = {(b["name"], b["kind"]) for b in backend.bindings}
        assert ("x", "value") in kinds
        assert ("z", "error") in kinds
        assert any(kind == "hole" for _, kind in kinds)
        # Supersession is visible in the persisted view: x v1 → superseded by 2.
        assert persisted[("x", 1)]["superseded_by"] == 2
        assert persisted[("x", 2)]["superseded_by"] is None

        # (5) Identity is stamped on every persisted dict.
        assert all(b["session_id"] == "S1" for b in backend.bindings)
        assert all(b["document_id"] == "D1" for b in backend.bindings)
        assert all(e["session_id"] == "S1" for e in backend.entries)
        assert all(e["document_id"] == "D1" for e in backend.entries)

        # (6) Everything the backend received is JSON-ready.
        json.dumps(backend.bindings)
