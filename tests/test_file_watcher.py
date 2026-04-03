"""Tests for sweep_local.file_watcher module."""

from sweep_local.file_watcher import DiffStore


def test_seed_file_no_diffs():
    store = DiffStore(max_diffs=5)
    store.seed_file("test.py", "initial content")
    diffs = store.get_recent_diffs("test.py")
    assert diffs == []


def test_update_file_produces_diffs():
    store = DiffStore(max_diffs=5)
    store.seed_file("test.py", "line1\nline2\n")
    diffs = store.update_file("test.py", "line1\nchanged\n")
    assert len(diffs) > 0
    assert diffs[0].file_path == "test.py"
    assert "line2" in diffs[0].old_chunk
    assert "changed" in diffs[0].new_chunk


def test_update_file_no_change():
    store = DiffStore(max_diffs=5)
    store.seed_file("test.py", "same\n")
    diffs = store.update_file("test.py", "same\n")
    assert diffs == []


def test_update_file_without_seed():
    store = DiffStore(max_diffs=5)
    # First update with no prior content should produce no diffs
    diffs = store.update_file("test.py", "new content\n")
    assert diffs == []


def test_get_recent_diffs_filters_by_file():
    store = DiffStore(max_diffs=5)
    store.seed_file("a.py", "old\n")
    store.seed_file("b.py", "old\n")
    store.update_file("a.py", "new\n")
    store.update_file("b.py", "new\n")
    a_diffs = store.get_recent_diffs("a.py")
    b_diffs = store.get_recent_diffs("b.py")
    assert all(d.file_path == "a.py" for d in a_diffs)
    assert all(d.file_path == "b.py" for d in b_diffs)


def test_get_recent_diffs_all():
    store = DiffStore(max_diffs=5)
    store.seed_file("a.py", "old\n")
    store.seed_file("b.py", "old\n")
    store.update_file("a.py", "new\n")
    store.update_file("b.py", "new\n")
    all_diffs = store.get_recent_diffs()
    assert len(all_diffs) >= 2


def test_max_diffs_limit():
    store = DiffStore(max_diffs=2)
    store.seed_file("test.py", "v0\n")
    store.update_file("test.py", "v1\n")
    store.update_file("test.py", "v2\n")
    store.update_file("test.py", "v3\n")
    diffs = store.get_recent_diffs("test.py")
    assert len(diffs) <= 2


def test_get_cached_content():
    store = DiffStore(max_diffs=5)
    assert store.get_cached_content("test.py") is None
    store.seed_file("test.py", "content")
    assert store.get_cached_content("test.py") == "content"
    store.update_file("test.py", "updated")
    assert store.get_cached_content("test.py") == "updated"


def test_multiline_diff():
    store = DiffStore(max_diffs=5)
    old = "line1\nline2\nline3\nline4\n"
    new = "line1\nNEW2\nNEW3\nline4\n"
    store.seed_file("test.py", old)
    diffs = store.update_file("test.py", new)
    assert len(diffs) > 0
