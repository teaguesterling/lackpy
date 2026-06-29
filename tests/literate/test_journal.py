"""Unit tests for the FileJournal (transactional file rollback)."""


from lackpy.interpreters.literate.journal import FileJournal


def test_rollback_restores_overwritten_content(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("original")
    j = FileJournal(tmp_path)
    j.snapshot(["a.txt"])
    f.write_text("mutated")          # simulate a write-cell
    j.rollback()
    assert f.read_text() == "original"


def test_rollback_deletes_files_created_after_snapshot(tmp_path):
    j = FileJournal(tmp_path)
    j.snapshot(["new.txt"])          # did not exist at snapshot time
    (tmp_path / "new.txt").write_text("created")
    j.rollback()
    assert not (tmp_path / "new.txt").exists()


def test_commit_drops_snapshots_so_rollback_is_a_noop(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v1")
    j = FileJournal(tmp_path)
    j.snapshot(["a.txt"])
    f.write_text("v2")
    j.commit()
    assert j.tracked == 0
    j.rollback()                     # nothing tracked -> must not resurrect v1
    assert f.read_text() == "v2"


def test_snapshot_is_idempotent_per_path(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("first")
    j = FileJournal(tmp_path)
    j.snapshot(["a.txt"])
    f.write_text("second")
    j.snapshot(["a.txt"])            # must NOT re-snapshot the mutated content
    assert j.tracked == 1
    j.rollback()
    assert f.read_text() == "first"


def test_relative_paths_resolve_against_base_dir(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "m.py").write_text("orig")
    j = FileJournal(tmp_path)
    j.snapshot(["src/m.py"])
    (sub / "m.py").write_text("changed")
    j.rollback()
    assert (sub / "m.py").read_text() == "orig"


def test_rollback_recreates_parent_dir_if_removed(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir()
    f = sub / "x.txt"
    f.write_text("keep")
    j = FileJournal(tmp_path)
    j.snapshot(["pkg/x.txt"])
    f.unlink()                       # cell deleted the file (and we drop the dir)
    sub.rmdir()
    j.rollback()
    assert f.read_text() == "keep"


def test_rollback_skips_a_directory_sharing_the_path(tmp_path):
    # If a tracked path is a directory at snapshot (None recorded), rollback must
    # not unlink it (IsADirectoryError) and abort the rest of the restore.
    d = tmp_path / "adir"
    d.mkdir()
    other = tmp_path / "b.txt"
    other.write_text("orig")
    j = FileJournal(tmp_path)
    j.snapshot(["adir", "b.txt"])
    other.write_text("changed")
    j.rollback()                     # must not raise on the directory entry
    assert other.read_text() == "orig"
    assert d.is_dir()
