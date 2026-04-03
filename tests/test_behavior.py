"""
Behavioral tests to characterize and verify current system behavior.

These tests exercise the interaction between components — prompt construction,
diff context, PSI context, n-gram speculation, token healing, and early stop —
to document how the system actually behaves in realistic scenarios.
"""

import mlx.core as mx

from sweep_local.file_watcher import DiffStore
from sweep_local.server import (
    PsiDefinition,
    _check_early_stop,
    build_ngram_index,
    ngram_draft_tokens,
    psi_to_chunks,
)
from sweep_local.sweep_prompt import (
    FileChunk,
    build_prompt,
    compute_prefill,
    format_diff,
    is_pure_insertion_above_cursor,
)
from sweep_local.token_healing import VocabTrie, find_healing_boundary, make_healing_processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REALISTIC_FILE = """\
import os
import sys
from pathlib import Path

class Config:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    def validate(self):
        if self.port < 0 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")

class Server:
    def __init__(self, config: Config):
        self.config = config
        self._running = False

    def start(self):
        self.config.validate()
        self._running = True
        print(f"Listening on {self.config.host}:{self.config.port}")

    def stop(self):
        self._running = False

    def handle_request(self, path: str, method: str = "GET"):
        if not self._running:
            raise RuntimeError("Server not running")
        return {"path": path, "method": method, "status": 200}

def main():
    config = Config()
    server = Server(config)
    server.start()
"""


class MockTokenizer:
    """Minimal greedy tokenizer for behavior tests."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self._id_to_text = {v: k for k, v in vocab.items()}
        self.bos_token_id = None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        tokens = []
        i = 0
        while i < len(text):
            best = None
            best_len = 0
            for t in self._vocab:
                if text[i:].startswith(t) and len(t) > best_len:
                    best = t
                    best_len = len(t)
            if best is None:
                raise ValueError(f"Cannot encode character: {text[i]!r}")
            tokens.append(self._vocab[best])
            i += best_len
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_text[i] for i in ids)


VOCAB = {
    "s": 0, "e": 1, "r": 2, "v": 3, "er": 4, "serv": 5, "server": 6,
    ".": 7, "start": 8, "(": 9, ")": 10, "\n": 11, " ": 12,
    "se": 13, "rver": 14, "st": 15, "art": 16, "a": 17, "t": 18,
    "self": 19, "config": 20, "hand": 21, "le": 22, "_": 23,
}


# ---------------------------------------------------------------------------
# 1. Prompt construction behavior
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify how prompts are built from file contents and cursor position."""

    def test_cursor_marker_is_at_exact_position(self):
        cursor = REALISTIC_FILE.index("server.start()")
        prompt, *_ = build_prompt("main.py", REALISTIC_FILE, cursor)
        # The cursor marker should appear in the prompt
        assert "<|cursor|>" in prompt
        # The text immediately after <|cursor|> should be "server.start()"
        marker_pos = prompt.index("<|cursor|>")
        after_marker = prompt[marker_pos + len("<|cursor|>"):]
        assert after_marker.startswith("server.start()")

    def test_code_block_window_is_centered_on_cursor(self):
        cursor = REALISTIC_FILE.index("self._running = True")
        _, code_block, block_start, relative_cursor = build_prompt(
            "main.py", REALISTIC_FILE, cursor, num_lines_before=3, num_lines_after=3,
        )
        lines = code_block.splitlines()
        # Should have roughly 7 lines (3 before + cursor line + 3 after)
        assert 4 <= len(lines) <= 8
        # The cursor line should be in the code block
        assert "self._running = True" in code_block

    def test_code_block_at_file_start(self):
        cursor = 0  # very beginning
        _, code_block, block_start, relative_cursor = build_prompt(
            "main.py", REALISTIC_FILE, cursor, num_lines_before=5, num_lines_after=5,
        )
        assert block_start == 0
        assert relative_cursor == 0
        assert code_block.startswith("import os")

    def test_code_block_at_file_end(self):
        cursor = len(REALISTIC_FILE) - 1
        _, code_block, _, relative_cursor = build_prompt(
            "main.py", REALISTIC_FILE, cursor, num_lines_before=5, num_lines_after=5,
        )
        # Should include the last lines
        assert "server.start()" in code_block

    def test_prompt_template_structure(self):
        cursor = REALISTIC_FILE.index("self.config = config")
        prompt, *_ = build_prompt("main.py", REALISTIC_FILE, cursor)
        # Verify the prompt has the expected sections
        assert "<|file_sep|>main.py" in prompt
        assert "<|file_sep|>original/main.py:" in prompt
        assert "<|file_sep|>current/main.py:" in prompt
        assert "<|file_sep|>updated/main.py:" in prompt

    def test_prev_section_matches_code_block(self):
        """The 'original' section should be the code block without cursor marker."""
        cursor = REALISTIC_FILE.index("self._running = False")
        prompt, code_block, *_ = build_prompt(
            "main.py", REALISTIC_FILE, cursor, num_lines_before=2, num_lines_after=2,
        )
        # original section should contain the exact code block
        original_marker = "<|file_sep|>original/main.py:"
        current_marker = "<|file_sep|>current/main.py:"
        orig_start = prompt.index(original_marker) + len(original_marker)
        # skip the ":start:end\n" part
        orig_start = prompt.index("\n", orig_start) + 1
        orig_end = prompt.index(current_marker)
        original_section = prompt[orig_start:orig_end].strip()
        assert original_section == code_block.strip()


# ---------------------------------------------------------------------------
# 2. Prefill behavior
# ---------------------------------------------------------------------------


class TestPrefillBehavior:
    """Verify how prefill works in different editing scenarios."""

    def test_prefill_excludes_cursor_line_in_default_mode(self):
        code = "line1\nline2\nline3\nline4\n"
        cursor = len("line1\nline2\n")  # cursor at start of line3
        prefill = compute_prefill(code, cursor, changes_above_cursor=False)
        assert "line1\n" in prefill
        assert "line2\n" in prefill
        assert "line3" not in prefill

    def test_prefill_changes_above_only_keeps_first_line(self):
        code = "first\nsecond\nthird\n"
        cursor = len("first\nsecond\n")
        prefill = compute_prefill(code, cursor, changes_above_cursor=True)
        assert "first" in prefill
        assert "second" not in prefill

    def test_prefill_empty_when_cursor_on_first_line(self):
        code = "only_line\n"
        cursor = 3
        prefill = compute_prefill(code, cursor, changes_above_cursor=False)
        assert prefill == ""

    def test_prefill_preserves_indentation(self):
        code = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
        cursor = len("def foo():\n    x = 1\n    y = 2\n")
        prefill = compute_prefill(code, cursor, changes_above_cursor=False)
        assert "    x = 1\n" in prefill
        assert "    y = 2\n" in prefill


# ---------------------------------------------------------------------------
# 3. Diff context behavior
# ---------------------------------------------------------------------------


class TestDiffContext:
    """Verify how file diffs are tracked and formatted for context."""

    def test_sequential_edits_produce_sequential_diffs(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("app.py", "v1\nline2\n")
        d1 = store.update_file("app.py", "v2\nline2\n")
        d2 = store.update_file("app.py", "v3\nline2\n")
        assert len(d1) == 1
        assert len(d2) == 1
        all_diffs = store.get_recent_diffs("app.py")
        assert len(all_diffs) == 2
        assert all_diffs[0].new_chunk.strip() == "v2"
        assert all_diffs[1].new_chunk.strip() == "v3"

    def test_multiline_edit_captures_full_change(self):
        store = DiffStore(max_diffs=10)
        old = "a\nb\nc\nd\ne\n"
        new = "a\nX\nY\nZ\ne\n"
        store.seed_file("f.py", old)
        diffs = store.update_file("f.py", new)
        assert len(diffs) >= 1
        # The diff should capture the changed region
        combined_old = "".join(d.old_chunk for d in diffs)
        combined_new = "".join(d.new_chunk for d in diffs)
        assert "b" in combined_old
        assert "X" in combined_new

    def test_insert_only_diff(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "a\nb\n")
        diffs = store.update_file("f.py", "a\nNEW\nb\n")
        assert len(diffs) >= 1

    def test_delete_only_diff(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "a\nDELETE_ME\nb\n")
        diffs = store.update_file("f.py", "a\nb\n")
        assert len(diffs) >= 1
        assert any("DELETE_ME" in d.old_chunk for d in diffs)

    def test_cross_file_diffs_are_sorted_by_time(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("a.py", "old\n")
        store.seed_file("b.py", "old\n")
        store.seed_file("c.py", "old\n")
        store.update_file("a.py", "new_a\n")
        store.update_file("b.py", "new_b\n")
        store.update_file("c.py", "new_c\n")
        all_diffs = store.get_recent_diffs()
        timestamps = [d.timestamp for d in all_diffs]
        assert timestamps == sorted(timestamps)

    def test_diff_format_includes_line_numbers(self):
        result = format_diff("test.py", 10, 15, "old code\n", "new code\n")
        assert "test.py:10:15" in result

    def test_seed_then_seed_overwrites_cache(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "first")
        store.seed_file("f.py", "second")
        assert store.get_cached_content("f.py") == "second"
        # No diffs should be generated from seed
        assert store.get_recent_diffs("f.py") == []

    def test_diffs_included_in_prompt(self):
        changes = format_diff("other.py", 1, 3, "old_func()\n", "new_func()\n")
        cursor = REALISTIC_FILE.index("server.start()")
        prompt, *_ = build_prompt(
            "main.py", REALISTIC_FILE, cursor, recent_changes=changes,
        )
        assert "old_func()" in prompt
        assert "new_func()" in prompt


# ---------------------------------------------------------------------------
# 4. PSI context behavior
# ---------------------------------------------------------------------------


class TestPsiContextBehavior:
    """Verify PSI definitions flow correctly into prompts."""

    def test_psi_definitions_appear_in_prompt(self):
        defs = [
            PsiDefinition(
                kind="class", name="Config",
                qualified_name="app.Config",
                signature="class Config:\n    host: str\n    port: int",
                file_path="app/config.py",
                priority=10,
            ),
        ]
        chunks = psi_to_chunks(defs)
        cursor = REALISTIC_FILE.index("config = Config()")
        prompt, *_ = build_prompt(
            "main.py", REALISTIC_FILE, cursor, retrieval_chunks=chunks,
        )
        assert "class Config:" in prompt
        assert "host: str" in prompt

    def test_psi_with_diffs_both_appear(self):
        defs = [
            PsiDefinition(
                kind="method", name="validate",
                qualified_name="Config.validate",
                signature="def validate(self) -> None:",
                priority=5,
            ),
        ]
        chunks = psi_to_chunks(defs)
        changes = format_diff("config.py", 5, 6, "old\n", "new\n")
        cursor = REALISTIC_FILE.index("self.config.validate()")
        prompt, *_ = build_prompt(
            "main.py", REALISTIC_FILE, cursor,
            recent_changes=changes, retrieval_chunks=chunks,
        )
        # Both PSI and diff context should be in the prompt
        assert "def validate" in prompt
        assert "old" in prompt and "new" in prompt

    def test_psi_priority_ordering_in_prompt(self):
        defs = [
            PsiDefinition(kind="class", name="Low", qualified_name="a.Low",
                           signature="class Low: ...", priority=1),
            PsiDefinition(kind="class", name="High", qualified_name="a.High",
                           signature="class High: ...", priority=10),
        ]
        chunks = psi_to_chunks(defs)
        cursor = REALISTIC_FILE.index("server = Server(config)")
        prompt, *_ = build_prompt(
            "main.py", REALISTIC_FILE, cursor, retrieval_chunks=chunks,
        )
        # Both should appear since they're small
        assert "class High:" in prompt
        assert "class Low:" in prompt
        # High priority should appear first in the prompt
        high_pos = prompt.index("class High:")
        low_pos = prompt.index("class Low:")
        assert high_pos < low_pos

    def test_empty_psi_context_no_effect(self):
        cursor = REALISTIC_FILE.index("server.start()")
        prompt_no_psi, *_ = build_prompt("main.py", REALISTIC_FILE, cursor)
        prompt_empty_psi, *_ = build_prompt(
            "main.py", REALISTIC_FILE, cursor, retrieval_chunks=[],
        )
        # Empty list should behave the same as None (no retrieval results added)
        # Actually, empty list is falsy so retrieval_results stays ""
        assert prompt_no_psi == prompt_empty_psi


# ---------------------------------------------------------------------------
# 5. N-gram speculation behavior
# ---------------------------------------------------------------------------


class TestNgramBehavior:
    """Verify n-gram speculative decoding patterns."""

    def test_repeated_pattern_uses_last_occurrence(self):
        # When a pattern appears multiple times, n-gram uses the LAST position
        # in the index, which is the last occurrence in the prompt.
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 7, 8]
        index = build_ngram_index(tokens, n=3)
        # (1, 2) → positions [2, 7]. positions[-1] = 7, so draft = tokens[7:9]
        draft = ngram_draft_tokens(index, tokens, [1, 2], num_draft=2, n=3)
        # tokens[7] = 3, tokens[8] = 7 → draft is what follows the LAST (1,2) bigram
        assert draft == [3, 7]

    def test_longer_context_preferred_over_shorter(self):
        tokens = [10, 20, 30, 40, 50, 60, 70, 80]
        index = build_ngram_index(tokens, n=4)
        # Context (20, 30, 40) should be tried before (30, 40) or (40,)
        draft = ngram_draft_tokens(index, tokens, [20, 30, 40], num_draft=2, n=4)
        # Trigram match: (20,30,40) → pos 4, so draft = [50, 60]
        assert draft == [50, 60]

    def test_ngram_draft_returns_partial_when_near_end(self):
        tokens = [1, 2, 3, 4, 5]
        index = build_ngram_index(tokens, n=3)
        # (3, 4) → position 4, only token 5 available for draft
        draft = ngram_draft_tokens(index, tokens, [3, 4], num_draft=5, n=3)
        assert draft == [5]  # only one token available

    def test_early_stop_consecutive_matching(self):
        """Verify the early stop mechanism checks consecutive suffix matches."""
        suffix = [100, 200, 300, 400, 500]
        # Simulate a sequence of generated tokens matching the suffix
        for i, tok in enumerate([100, 200, 300, 400, 500]):
            assert _check_early_stop(tok, suffix, i) is True
        # A mismatch at any point returns False
        assert _check_early_stop(999, suffix, 0) is False

    def test_early_stop_resets_on_mismatch(self):
        """Document that the caller is responsible for resetting suffix_pos on mismatch."""
        suffix = [100, 200, 300]
        # Match first two
        assert _check_early_stop(100, suffix, 0) is True
        assert _check_early_stop(200, suffix, 1) is True
        # Mismatch — caller should reset pos to 0
        assert _check_early_stop(999, suffix, 2) is False


# ---------------------------------------------------------------------------
# 6. Token healing behavior
# ---------------------------------------------------------------------------


class TestTokenHealingBehavior:
    """Verify token healing in realistic scenarios."""

    def test_stable_boundary_no_healing(self):
        tok = MockTokenizer(VOCAB)
        # "server" encodes as [6] — single token, stable
        tokens = tok.encode("server")
        trimmed, prefix = find_healing_boundary(tokens, tok)
        assert prefix == ""
        assert trimmed == tokens

    def test_healing_processor_masks_all_but_valid_continuations(self):
        tok = MockTokenizer(VOCAB)
        trie = VocabTrie.from_tokenizer(tok)
        vocab_size = len(VOCAB)

        processor = make_healing_processor("er", trie, tok, prompt_token_count=5)
        prompt_tokens = mx.array([0, 1, 2, 3, 4])
        logits = mx.zeros(vocab_size)

        result = processor(prompt_tokens, logits)
        result_list = result.tolist()

        # "er" (4) should be allowed (exact match, starts with "er")
        allowed = trie.continuation_search("er")
        for tid in range(vocab_size):
            if tid in allowed:
                assert result_list[tid] == 0.0
            else:
                assert result_list[tid] == float("-inf")

    def test_healing_processor_tracks_multi_token_consumption(self):
        tok = MockTokenizer(VOCAB)
        trie = VocabTrie.from_tokenizer(tok)
        vocab_size = len(VOCAB)

        # Prefix "start" should be consumed in steps
        processor = make_healing_processor("start", trie, tok, prompt_token_count=3)
        prompt = mx.array([7, 7, 7])
        logits = mx.zeros(vocab_size)

        # Step 1: constrain for "start"
        r1 = processor(prompt, logits)
        # "start" (8) should be allowed
        assert r1.tolist()[8] == 0.0

        # Step 2: "start" was generated → prefix fully consumed → no-op
        tokens2 = mx.array([7, 7, 7, 8])  # prompt + "start"
        r2 = processor(tokens2, logits)
        # Should be no-op (all zeros, no -inf)
        assert all(v == 0.0 for v in r2.tolist())


# ---------------------------------------------------------------------------
# 7. Pure insertion rejection behavior
# ---------------------------------------------------------------------------


class TestPureInsertionRejection:
    """Verify that completions inserting only above the cursor are rejected."""

    def test_insertion_above_blank_cursor_line_not_rejected(self):
        # When cursor is on a blank line, pure insertion check returns False
        # because blank cursor lines are excluded (cursor_line.strip() == "")
        code = "import os\n\ndef main():\n    pass\n"
        cursor = len("import os\n")  # cursor on empty line between imports and def
        completion = "import os\nimport sys\n\ndef main():\n    pass\n"
        assert is_pure_insertion_above_cursor(code, completion, cursor) is False

    def test_insertion_above_non_blank_cursor_line_rejected(self):
        # Note: cursor position len("import os\n") is treated as END of line 1
        # ("import os\n"), not beginning of line 2. The function uses
        # current_line_index = len(code[:cursor].splitlines(True)).
        # To have cursor on "def main():", we need to be within that line.
        code = "import os\ndef main():\n    pass\n"
        cursor = len("import os\n") + 1  # cursor within "def main():" line
        completion = "import os\nimport sys\ndef main():\n    pass\n"
        # prefix="import os\n", cursor_line="def main():\n", suffix="    pass\n"
        # completion ends with "def main():\n    pass\n" → True
        assert is_pure_insertion_above_cursor(code, completion, cursor) is True

    def test_edit_at_cursor_line_accepted(self):
        code = "import os\n\ndef main():\n    pass\n"
        cursor = len("import os\n\ndef main():\n")  # cursor at "    pass"
        # Completion changes the cursor line
        completion = "import os\n\ndef main():\n    print('hello')\n"
        assert is_pure_insertion_above_cursor(code, completion, cursor) is False

    def test_no_rejection_when_whole_block_matches(self):
        code = "a\nb\nc\n"
        cursor = len("a\n")
        # Completion is identical to code block — not a pure insertion
        assert is_pure_insertion_above_cursor(code, code, cursor) is False

    def test_no_rejection_on_blank_cursor_line(self):
        code = "a\n\nc\n"
        cursor = len("a\n")  # cursor on empty line
        completion = "a\nINSERTED\n\nc\n"
        # Empty cursor line → should not be flagged
        assert is_pure_insertion_above_cursor(code, completion, cursor) is False


# ---------------------------------------------------------------------------
# 8. Adaptive context window behavior
# ---------------------------------------------------------------------------


class TestAdaptiveContext:
    """Verify that context window shrinks when PSI chunks are present."""

    def test_context_reduces_with_psi_chunks(self):
        # Build a file with many lines
        big_file = "\n".join(f"line_{i} = {i}" for i in range(500))
        cursor = big_file.index("line_250")

        prompt_no_psi, *_ = build_prompt("big.py", big_file, cursor)
        chunks = [FileChunk(file_path="defs/X", content="class X: pass")]
        prompt_with_psi, *_ = build_prompt("big.py", big_file, cursor, retrieval_chunks=chunks)

        # The prompt with PSI should be shorter (less broad context)
        assert len(prompt_with_psi) < len(prompt_no_psi)

    def test_context_still_has_minimum_when_psi_present(self):
        """Even with PSI, at least 25 lines of context in each direction."""
        big_file = "\n".join(f"line_{i} = {i}" for i in range(500))
        cursor = big_file.index("line_250")
        chunks = [FileChunk(file_path="defs/X", content="class X: pass")]
        prompt, *_ = build_prompt("big.py", big_file, cursor, retrieval_chunks=chunks)

        # The context should still contain lines near the cursor
        # With PSI, context_half=25 and chunk quantization at 20-line boundaries
        assert "line_230" in prompt
        assert "line_260" in prompt


# ---------------------------------------------------------------------------
# 9. End-to-end prompt scenarios
# ---------------------------------------------------------------------------


class TestEndToEndPromptScenarios:
    """Realistic scenarios combining multiple features."""

    def test_dot_access_scenario_with_psi(self):
        """Simulate typing 'server.' with PSI providing Server's methods."""
        file = "from app import Server\n\nserver = Server()\nserver.\n"
        cursor = file.index("server.\n") + len("server.")
        psi_chunks = [
            FileChunk(
                file_path="app/server.py",
                content="class Server:\n    def start(self): ...\n    def stop(self): ...\n    def handle_request(self, path: str): ...",
            ),
        ]
        prompt, code_block, _, relative_cursor = build_prompt(
            "main.py", file, cursor, retrieval_chunks=psi_chunks,
        )
        # Prompt should contain Server's methods for the model to use
        assert "def start" in prompt
        assert "def stop" in prompt
        assert "def handle_request" in prompt
        # Cursor should be right after "server."
        assert code_block[relative_cursor - 1] == "."

    def test_function_call_scenario_with_diffs_and_psi(self):
        """Simulate editing a function call with both diff context and PSI."""
        file = "from db import query\n\nresult = query(\n"
        cursor = len(file) - 1  # cursor at end of "query("
        changes = format_diff("db.py", 10, 12, "def query():\n", "def query(sql: str):\n")
        psi_chunks = [
            FileChunk(
                file_path="db.py",
                content="def query(sql: str, params: list = None) -> list:",
            ),
        ]
        prompt, *_ = build_prompt(
            "main.py", file, cursor,
            recent_changes=changes, retrieval_chunks=psi_chunks,
        )
        # Both context sources should be present
        assert "def query(sql: str, params: list" in prompt  # PSI
        assert "def query():" in prompt  # old diff
        assert "def query(sql: str):" in prompt  # new diff

    def test_empty_file_produces_empty_prompt(self):
        prompt, code_block, block_start, relative_cursor = build_prompt(
            "empty.py", "", 0
        )
        assert prompt == ""
        assert code_block == ""
        assert block_start == 0
        assert relative_cursor == 0

    def test_single_line_file(self):
        file = "x = 1"
        cursor = 2
        prompt, code_block, _, relative_cursor = build_prompt("f.py", file, cursor)
        assert "<|cursor|>" in prompt
        assert "x = 1" in code_block
        assert relative_cursor == 2
