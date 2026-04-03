"""Tests for sweep_local.sweep_prompt module."""

from sweep_local.sweep_prompt import (
    build_prompt,
    compute_prefill,
    format_diff,
    is_pure_insertion_above_cursor,
)

# --- format_diff ---


def test_format_diff_basic():
    result = format_diff(
        file_path="main.py",
        start_line=5,
        end_line=7,
        old_code="x = 1\ny = 2\n",
        new_code="x = 10\ny = 20\n",
    )
    assert "main.py:5:7" in result
    assert "x = 1" in result
    assert "x = 10" in result
    assert "original:" in result
    assert "updated:" in result


def test_format_diff_empty_old():
    result = format_diff("f.py", 1, 1, "", "new line\n")
    assert "new line" in result


# --- compute_prefill ---


def test_compute_prefill_default_mode():
    code_block = "line1\nline2\nline3\n"
    # Cursor at start of "line3" (after "line1\nline2\n")
    cursor = len("line1\nline2\n")
    prefill = compute_prefill(code_block, cursor, changes_above_cursor=False)
    assert prefill == "line1\nline2\n"


def test_compute_prefill_cursor_on_first_line():
    code_block = "hello world\nline2\n"
    cursor = 5  # middle of first line
    prefill = compute_prefill(code_block, cursor, changes_above_cursor=False)
    assert prefill == ""


def test_compute_prefill_changes_above():
    code_block = "line1\nline2\nline3\n"
    cursor = len("line1\nline2\n")
    prefill = compute_prefill(code_block, cursor, changes_above_cursor=True)
    # Should only include first line + leading newlines
    assert "line1" in prefill
    assert "line2" not in prefill or prefill.strip() == "line1"


# --- is_pure_insertion_above_cursor ---


def test_pure_insertion_above_detected():
    # code_block: 3 lines, cursor at start of line 3 ("ccc")
    code_block = "aaa\nbbb\nccc\n"
    cursor = len("aaa\nbbb\n")
    # Completion inserts a line between aaa and bbb, keeps ccc (cursor line) + rest
    # This IS a pure insertion above cursor -- the cursor line "ccc" is unchanged
    completion = "aaa\nINSERTED\nbbb\nccc\n"
    assert is_pure_insertion_above_cursor(code_block, completion, cursor) is True


def test_pure_insertion_above_exact():
    # A completion that exactly inserts above cursor line without touching it
    code_block = "aaa\nbbb\nccc\n"
    cursor = len("aaa\nbbb\n")  # cursor at "ccc"
    # prefix = "aaa\n", cursor_line = "bbb\n", suffix = "ccc\n"
    # Wait - let's trace the logic carefully:
    # current_line_index = number of lines in code_block[:cursor] = 2 (lines "aaa\n" and "bbb\n")
    # cursor_line = code_block_lines[1] = "bbb\n"
    # prefix_lines = code_block_lines[:1] = ["aaa\n"]  -> prefix = "aaa\n"
    # suffix_lines = code_block_lines[2:] = ["ccc\n"]  -> suffix = "ccc\n"
    # For pure insertion: completion.startswith("aaa\n") and completion.endswith("bbb\nccc\n")
    completion = "aaa\nNEW_LINE\nbbb\nccc\n"
    assert is_pure_insertion_above_cursor(code_block, completion, cursor) is True


def test_not_pure_insertion_when_cursor_line_changed():
    code_block = "aaa\nbbb\nccc\n"
    cursor = len("aaa\n")
    # Completion changes bbb to BBB -- not a pure insertion above
    completion = "aaa\nBBB\nccc\n"
    assert is_pure_insertion_above_cursor(code_block, completion, cursor) is False


def test_not_pure_insertion_empty_code():
    assert is_pure_insertion_above_cursor("", "something", 0) is False


def test_not_pure_insertion_zero_cursor():
    assert is_pure_insertion_above_cursor("abc\n", "abc\n", 0) is False


# --- build_prompt ---


SAMPLE_FILE = """\
import os
import sys

def hello():
    print("hello")

def world():
    print("world")

def main():
    hello()
    world()
"""


def test_build_prompt_returns_tuple():
    cursor = SAMPLE_FILE.index("print(\"hello\")")
    result = build_prompt(
        file_path="example.py",
        file_contents=SAMPLE_FILE,
        cursor_position=cursor,
    )
    assert isinstance(result, tuple)
    assert len(result) == 4
    prompt, code_block, block_start, relative_cursor = result
    assert isinstance(prompt, str)
    assert isinstance(code_block, str)
    assert isinstance(block_start, int)
    assert isinstance(relative_cursor, int)


def test_build_prompt_contains_file_path():
    cursor = SAMPLE_FILE.index("print(\"hello\")")
    prompt, *_ = build_prompt("example.py", SAMPLE_FILE, cursor)
    assert "example.py" in prompt


def test_build_prompt_contains_cursor_marker():
    cursor = SAMPLE_FILE.index("print(\"hello\")")
    prompt, *_ = build_prompt("example.py", SAMPLE_FILE, cursor)
    assert "<|cursor|>" in prompt


def test_build_prompt_with_recent_changes():
    cursor = SAMPLE_FILE.index("print(\"hello\")")
    changes = format_diff("example.py", 1, 2, "old\n", "new\n")
    prompt, *_ = build_prompt("example.py", SAMPLE_FILE, cursor, recent_changes=changes)
    assert "old" in prompt
    assert "new" in prompt


def test_build_prompt_empty_file():
    prompt, code_block, block_start, relative_cursor = build_prompt("empty.py", "", 0)
    assert prompt == ""
    assert code_block == ""


def test_build_prompt_cursor_at_end():
    cursor = len(SAMPLE_FILE)
    prompt, code_block, block_start, relative_cursor = build_prompt(
        "example.py", SAMPLE_FILE, cursor
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_respects_line_window():
    cursor = SAMPLE_FILE.index("print(\"hello\")")
    _, code_block, _, _ = build_prompt(
        "example.py", SAMPLE_FILE, cursor,
        num_lines_before=2, num_lines_after=2,
    )
    # Code block should be a subset, not the full file
    assert len(code_block) < len(SAMPLE_FILE)
