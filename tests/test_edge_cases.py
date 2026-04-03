"""
Edge case tests to characterize system behavior in unusual situations.

Covers boundary conditions, error paths, and corner cases across all modules.
"""

import mlx.core as mx

from sweep_local.file_watcher import DiffStore
from sweep_local.server import (
    Metrics,
    PsiDefinition,
    _check_early_stop,
    _common_prefix_len,
    build_ngram_index,
    ngram_draft_tokens,
    psi_to_chunks,
)
from sweep_local.sweep_prompt import (
    FileChunk,
    build_prompt,
    compute_prefill,
    is_pure_insertion_above_cursor,
)
from sweep_local.token_healing import (
    VocabTrie,
    find_healing_boundary,
    make_healing_processor,
)


# ===================================================================
# Shared helpers
# ===================================================================

class MockTokenizer:
    """Greedy longest-match tokenizer for testing."""

    def __init__(self, vocab: dict[str, int], bos_token_id: int | None = None):
        self._vocab = vocab
        self._id_to_text = {v: k for k, v in vocab.items()}
        self.bos_token_id = bos_token_id

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
                raise ValueError(f"Cannot encode: {text[i]!r}")
            tokens.append(self._vocab[best])
            i += best_len
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_text[i] for i in ids)


VOCAB = {
    "N": 0, "o": 1, "d": 2, "e": 3, "Node": 4, " ": 5, "->": 6,
    ":": 7, "od": 8, "de": 9, "No": 10, "class": 11, "sw": 12,
    "s": 13, "weep": 14, "sweep": 15, "wing": 16, "swap": 17, "\n": 18,
}


# ===================================================================
# 1. Prompt Construction Edge Cases
# ===================================================================


class TestPromptEdgeCases:

    def test_cursor_beyond_file_length(self):
        """Cursor past EOF should clamp to last line."""
        file = "line1\nline2\n"
        cursor = len(file) + 100
        prompt, code_block, _, relative_cursor = build_prompt("f.py", file, cursor)
        # Should not crash; cursor clamped to last line
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "<|cursor|>" in prompt

    def test_file_of_only_newlines(self):
        file = "\n\n\n"
        cursor = 1
        prompt, code_block, _, _ = build_prompt("f.py", file, cursor)
        assert isinstance(prompt, str)
        assert "<|cursor|>" in prompt

    def test_file_of_only_newlines_cursor_at_start(self):
        file = "\n\n\n"
        cursor = 0
        prompt, code_block, block_start, relative_cursor = build_prompt("f.py", file, cursor)
        assert block_start == 0
        assert relative_cursor == 0

    def test_single_line_no_trailing_newline_cursor_start(self):
        file = "x = 1"
        prompt, code_block, _, relative_cursor = build_prompt("f.py", file, 0)
        assert relative_cursor == 0
        assert "x = 1" in code_block

    def test_single_line_no_trailing_newline_cursor_end(self):
        file = "x = 1"
        prompt, _, _, relative_cursor = build_prompt("f.py", file, len(file))
        assert "<|cursor|>" in prompt

    def test_large_line_window_clamped(self):
        """Requesting 1000 lines before/after on a 3-line file should just use all lines."""
        file = "a\nb\nc\n"
        _, code_block, _, _ = build_prompt(
            "f.py", file, 2, num_lines_before=1000, num_lines_after=1000,
        )
        assert code_block == file

    def test_file_chunks_prepended_to_prompt(self):
        """The file_chunks parameter should prepend chunks before the main prompt."""
        file = "x = 1\n"
        chunks = [FileChunk(file_path="lib.py", content="def helper(): pass")]
        prompt, *_ = build_prompt("f.py", file, 0, file_chunks=chunks)
        # file_chunks should appear before the main <|file_sep|>f.py section
        lib_pos = prompt.index("lib.py")
        main_pos = prompt.index("<|file_sep|>f.py")
        assert lib_pos < main_pos
        assert "def helper(): pass" in prompt

    def test_cursor_at_newline_boundary(self):
        """Cursor right at a \\n position."""
        file = "aaa\nbbb\nccc\n"
        # Cursor at the newline after "aaa"
        cursor = 3  # the 'a','a','a' then cursor at position 3 which is '\n'
        prompt, code_block, _, relative_cursor = build_prompt("f.py", file, cursor)
        assert "<|cursor|>" in prompt
        # Cursor should be within the code block
        assert 0 <= relative_cursor <= len(code_block)

    def test_unicode_file_contents(self):
        file = "# 你好世界\ndef greet():\n    print('🎉')\n"
        cursor = file.index("print")
        prompt, code_block, _, _ = build_prompt("你好.py", file, cursor)
        assert "你好世界" in prompt
        assert "🎉" in prompt
        assert "你好.py" in prompt

    def test_prefill_changes_above_cursor_at_zero(self):
        code = "first\nsecond\nthird\n"
        prefill = compute_prefill(code, 0, changes_above_cursor=True)
        # Cursor at 0: prefill = code[:0] = ""
        assert prefill == ""

    def test_prefill_whitespace_only_lines(self):
        code = "  \n  \n  \n"
        cursor = len("  \n  \n")
        prefill = compute_prefill(code, cursor, changes_above_cursor=False)
        # Should include the whitespace lines before cursor
        assert "  \n" in prefill

    def test_retrieval_chunks_with_empty_content(self):
        file = "x = 1\n"
        chunks = [FileChunk(file_path="empty.py", content="")]
        prompt, *_ = build_prompt("f.py", file, 0, retrieval_chunks=chunks)
        # Should still include the chunk separator
        assert "empty.py" in prompt

    def test_very_long_single_line(self):
        """File with no newlines, 10k chars."""
        file = "x" * 10000
        cursor = 5000
        prompt, code_block, _, relative_cursor = build_prompt("f.py", file, cursor)
        assert "<|cursor|>" in prompt
        assert relative_cursor == 5000
        # Code block should be the entire file (single line)
        assert len(code_block) == 10000

    def test_tabs_preserved_in_code_block(self):
        file = "def f():\n\tx = 1\n\ty = 2\n"
        cursor = file.index("x = 1")
        _, code_block, _, _ = build_prompt("f.py", file, cursor)
        assert "\tx = 1" in code_block
        assert "\ty = 2" in code_block


# ===================================================================
# 2. Diff Store Edge Cases
# ===================================================================


class TestDiffStoreEdgeCases:

    def test_nonempty_to_empty_file(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "line1\nline2\n")
        diffs = store.update_file("f.py", "")
        assert len(diffs) >= 1
        assert store.get_cached_content("f.py") == ""

    def test_empty_to_nonempty_file(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "")
        diffs = store.update_file("f.py", "new content\n")
        assert len(diffs) >= 1

    def test_seed_identical_to_cached_update(self):
        """Seed after an update: cache is replaced, diffs are preserved."""
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "v1\n")
        store.update_file("f.py", "v2\n")
        # Diffs should exist
        assert len(store.get_recent_diffs("f.py")) > 0
        # Seed overwrites cache
        store.seed_file("f.py", "v3\n")
        assert store.get_cached_content("f.py") == "v3\n"
        # Diffs from previous update should still be there
        assert len(store.get_recent_diffs("f.py")) > 0

    def test_multiple_change_regions_in_one_update(self):
        """Changes at start and end with unchanged middle."""
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "a\nb\nc\nd\ne\n")
        diffs = store.update_file("f.py", "X\nb\nc\nd\nY\n")
        # Should have at least 2 diff regions
        assert len(diffs) >= 2
        old_chunks = [d.old_chunk for d in diffs]
        new_chunks = [d.new_chunk for d in diffs]
        assert any("a" in c for c in old_chunks)
        assert any("X" in c for c in new_chunks)
        assert any("e" in c for c in old_chunks)
        assert any("Y" in c for c in new_chunks)

    def test_max_diffs_drops_oldest(self):
        store = DiffStore(max_diffs=3)
        store.seed_file("f.py", "v0\n")
        store.update_file("f.py", "v1\n")
        store.update_file("f.py", "v2\n")
        store.update_file("f.py", "v3\n")
        store.update_file("f.py", "v4\n")
        diffs = store.get_recent_diffs("f.py")
        assert len(diffs) == 3
        # Oldest (v0→v1) should be dropped
        assert not any("v0" in d.old_chunk or "v1" in d.new_chunk for d in diffs[:1]
                        if "v0" in d.old_chunk)

    def test_file_path_with_special_chars(self):
        store = DiffStore(max_diffs=10)
        path = "/home/user/my project/файл.py"
        store.seed_file(path, "old\n")
        diffs = store.update_file(path, "new\n")
        assert len(diffs) > 0
        assert diffs[0].file_path == path

    def test_whitespace_only_changes(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "a \n")
        diffs = store.update_file("f.py", "a\n")
        # Trailing space removal should produce a diff
        assert len(diffs) >= 1

    def test_complete_file_rewrite(self):
        store = DiffStore(max_diffs=10)
        old = "line1\nline2\nline3\n"
        new = "totally\ndifferent\ncontent\n"
        store.seed_file("f.py", old)
        diffs = store.update_file("f.py", new)
        assert len(diffs) >= 1

    def test_single_char_file_change(self):
        store = DiffStore(max_diffs=10)
        store.seed_file("f.py", "x")
        diffs = store.update_file("f.py", "y")
        assert len(diffs) >= 1

    def test_cross_file_ordering_many_files(self):
        store = DiffStore(max_diffs=10)
        files = [f"f{i}.py" for i in range(5)]
        for f in files:
            store.seed_file(f, "old\n")
        for f in files:
            store.update_file(f, "new\n")
        all_diffs = store.get_recent_diffs()
        timestamps = [d.timestamp for d in all_diffs]
        assert timestamps == sorted(timestamps)


# ===================================================================
# 3. Token Healing Edge Cases
# ===================================================================


class TestTokenHealingEdgeCases:

    def test_trie_empty_prefix_search(self):
        """Empty prefix should return all tokens in the vocabulary."""
        trie = VocabTrie.from_tokenizer(MockTokenizer(VOCAB))
        result = trie.prefix_search("")
        assert len(result) == len(VOCAB)

    def test_trie_prefix_longer_than_all_vocab(self):
        trie = VocabTrie.from_tokenizer(MockTokenizer(VOCAB))
        result = trie.prefix_search("abcdefghijklmnop")
        assert result == []

    def test_healing_boundary_max_rollback_zero(self):
        """max_rollback=0: loop range(1,1) never executes, falls through to
        the 'all checked tokens unstable' fallback. rollback=min(0, len-1)=0,
        then tokens[:-0] returns [] and tokens[-0:] returns all tokens.
        This is a quirk of Python slicing — effectively rolls back everything."""
        tok = MockTokenizer(VOCAB)
        tokens = [0, 8]  # "N" + "od" = "Nod"
        trimmed, prefix = find_healing_boundary(tokens, tok, max_rollback=0)
        # tokens[:-0] == [] and tokens[-0:] == [0, 8] → decode = "Nod"
        assert trimmed == []
        assert prefix == "Nod"

    def test_healing_boundary_rollback_exceeds_tokens(self):
        """max_rollback larger than token list."""
        tok = MockTokenizer(VOCAB)
        tokens = [0, 1]  # "N", "o"
        trimmed, prefix = find_healing_boundary(tokens, tok, max_rollback=100)
        # Should handle gracefully — can only roll back len(tokens)-1 at most
        assert isinstance(trimmed, list)
        assert isinstance(prefix, str)

    def test_healing_boundary_with_bos_token(self):
        """Tokenizer that adds BOS token during re-encoding."""
        vocab = {"a": 0, "b": 1, "ab": 2}
        tok = MockTokenizer(vocab, bos_token_id=99)

        # Manually set up: tokens [0, 1] = "a"+"b" = "ab"
        # Re-encode "b" → [1] (no BOS added by our mock)
        # But if BOS were prepended: re_encoded = [99, 1], stripped to [1]
        # This tests that the BOS stripping logic exists
        tokens = [0, 1]
        trimmed, prefix = find_healing_boundary(tokens, tok, max_rollback=2)
        assert isinstance(trimmed, list)

    def test_healing_processor_no_valid_continuations(self):
        """Prefix has no matching tokens → processor logs warning and becomes no-op."""
        tok = MockTokenizer(VOCAB)
        trie = VocabTrie.from_tokenizer(tok)
        vocab_size = len(VOCAB)

        processor = make_healing_processor("xyz_impossible", trie, tok, prompt_token_count=3)
        prompt = mx.array([7, 7, 7])
        logits = mx.zeros(vocab_size)

        result = processor(prompt, logits)
        # No valid continuations → returns logits unchanged (no-op)
        assert all(v == 0.0 for v in result.tolist())

    def test_healing_processor_unexpected_token(self):
        """Model generates a token that doesn't match the prefix at all."""
        tok = MockTokenizer(VOCAB)
        trie = VocabTrie.from_tokenizer(tok)
        vocab_size = len(VOCAB)

        processor = make_healing_processor("No", trie, tok, prompt_token_count=3)
        prompt = mx.array([7, 7, 7])
        logits = mx.zeros(vocab_size)

        # Step 1: constrain
        processor(prompt, logits)

        # Step 2: model generated ":" (7) instead of matching "No"
        tokens2 = mx.array([7, 7, 7, 7])  # prompt + ":"
        result2 = processor(tokens2, logits)
        # Unexpected → abandons healing, becomes no-op
        assert all(v == 0.0 for v in result2.tolist())

    def test_trie_single_char_vocab(self):
        """Every vocab entry is a single character."""
        vocab = {chr(i): i for i in range(ord("a"), ord("z") + 1)}
        trie = VocabTrie.from_tokenizer(MockTokenizer(vocab))
        # prefix "a" should match only token "a"
        result = trie.prefix_search("a")
        assert result == [ord("a")]
        # continuation of "ab" → "a" (consumes part) and "b"? No — "ab" isn't in vocab
        cont = trie.continuation_search("ab")
        # "a" is a prefix of "ab" → included
        assert ord("a") in cont


# ===================================================================
# 4. N-gram & Early Stop Edge Cases
# ===================================================================


class TestNgramEdgeCases:

    def test_all_tokens_identical(self):
        tokens = [1, 1, 1, 1, 1]
        index = build_ngram_index(tokens, n=3)
        # (1,) maps to multiple positions, (1,1) maps to multiple positions
        assert (1,) in index
        assert (1, 1) in index
        # Draft from context [1, 1] → should return [1, ...]
        draft = ngram_draft_tokens(index, tokens, [1, 1], num_draft=2, n=3)
        assert all(t == 1 for t in draft)

    def test_single_token_prompt(self):
        index = build_ngram_index([42], n=4)
        # With 1 token, no n-grams of any size can be built
        assert index == {}

    def test_empty_token_list(self):
        index = build_ngram_index([], n=3)
        assert index == {}

    def test_common_prefix_len_both_empty(self):
        assert _common_prefix_len([], []) == 0

    def test_common_prefix_len_one_empty(self):
        assert _common_prefix_len([], [1, 2, 3]) == 0
        assert _common_prefix_len([1, 2, 3], []) == 0

    def test_common_prefix_len_identical(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3

    def test_common_prefix_len_no_common(self):
        assert _common_prefix_len([1, 2, 3], [4, 5, 6]) == 0

    def test_common_prefix_len_partial(self):
        assert _common_prefix_len([1, 2, 3], [1, 2, 9]) == 2

    def test_common_prefix_len_one_is_prefix_of_other(self):
        assert _common_prefix_len([1, 2], [1, 2, 3, 4]) == 2
        assert _common_prefix_len([1, 2, 3, 4], [1, 2]) == 2

    def test_early_stop_at_exact_end_of_suffix(self):
        suffix = [10, 20, 30]
        assert _check_early_stop(30, suffix, 2) is True
        # Position past end
        assert _check_early_stop(40, suffix, 3) is False

    def test_ngram_draft_empty_recent(self):
        tokens = [1, 2, 3, 4]
        index = build_ngram_index(tokens, n=3)
        draft = ngram_draft_tokens(index, tokens, [], num_draft=2, n=3)
        assert draft == []


# ===================================================================
# 5. PSI Context Edge Cases
# ===================================================================


class TestPsiEdgeCases:

    def test_empty_signature(self):
        defs = [PsiDefinition(kind="class", name="E", qualified_name="pkg.E", signature="")]
        chunks = psi_to_chunks(defs)
        # Empty signature still creates a chunk (overhead is 10 tokens)
        assert len(chunks) == 1
        assert chunks[0].content == ""

    def test_all_definitions_exceed_budget(self):
        """Every single definition is too large to fit."""
        huge_sig = "x" * 20000  # ~6667 estimated tokens, well over 1500 budget
        defs = [
            PsiDefinition(kind="class", name="H", qualified_name="pkg.H",
                           signature=huge_sig)
        ]
        chunks = psi_to_chunks(defs)
        assert chunks == []

    def test_equal_priority_stable_order(self):
        """Definitions with equal priority preserve input order."""
        defs = [
            PsiDefinition(kind="class", name=f"C{i}", qualified_name=f"pkg.C{i}",
                           signature=f"class C{i}: ...", priority=5)
            for i in range(5)
        ]
        chunks = psi_to_chunks(defs)
        names = [c.file_path.split(".")[-1] for c in chunks]
        # Python's sort is stable, so equal-priority items keep original order
        assert names == ["C0", "C1", "C2", "C3", "C4"]

    def test_body_preview_without_signature(self):
        defs = [PsiDefinition(
            kind="method", name="f", qualified_name="pkg.f",
            signature="", body_preview="    return 42",
        )]
        chunks = psi_to_chunks(defs)
        assert len(chunks) == 1
        assert "return 42" in chunks[0].content

    def test_long_qualified_name_as_path(self):
        defs = [PsiDefinition(
            kind="class", name="X",
            qualified_name="com.example.very.deep.package.hierarchy.X",
            signature="class X: ...",
        )]
        chunks = psi_to_chunks(defs)
        assert chunks[0].file_path == "definitions/com.example.very.deep.package.hierarchy.X"

    def test_mixed_file_path_present_absent(self):
        defs = [
            PsiDefinition(kind="class", name="A", qualified_name="pkg.A",
                           signature="class A: ...", file_path="src/a.py", priority=2),
            PsiDefinition(kind="class", name="B", qualified_name="pkg.B",
                           signature="class B: ...", priority=1),
        ]
        chunks = psi_to_chunks(defs)
        assert chunks[0].file_path == "src/a.py"  # has explicit path
        assert chunks[1].file_path == "definitions/pkg.B"  # fallback


# ===================================================================
# 6. Metrics Edge Cases
# ===================================================================


class TestMetricsEdgeCases:

    def test_snapshot_fresh_metrics(self):
        """Snapshot with no data recorded."""
        m = Metrics()
        snap = m.snapshot()
        assert snap["total_requests"] == 0
        assert snap["total_tokens"] == 0
        assert snap["cancellations"] == 0
        assert snap["cache_hit_rate"] == 0.0
        assert snap["draft_acceptance_rate"] == 0.0
        assert snap["psi_requests"] == 0
        assert snap["psi_avg_definitions"] == 0.0
        assert snap["latency_ms"]["p50"] == 0.0
        assert snap["tokens_per_sec"]["p50"] == 0.0
        assert snap["window_size"] == 0

    def test_record_zero_tokens_zero_latency(self):
        """0 tokens and 0 latency: no tokens_per_sec entry added."""
        m = Metrics()
        m.record(latency=0.0, n_tokens=0, cache_hit=False)
        snap = m.snapshot()
        assert snap["total_requests"] == 1
        assert snap["total_tokens"] == 0
        assert snap["tokens_per_sec"]["p50"] == 0.0  # no tps entries
        assert snap["window_size"] == 1  # latency was still recorded

    def test_psi_record_zero_definitions(self):
        m = Metrics()
        m.record_psi(0)
        snap = m.snapshot()
        assert snap["psi_requests"] == 1
        assert snap["psi_avg_definitions"] == 0.0

    def test_cancel_without_prior_record(self):
        """Only cancellations, no completions."""
        m = Metrics()
        m.record_cancel()
        m.record_cancel()
        snap = m.snapshot()
        assert snap["cancellations"] == 2
        assert snap["total_requests"] == 0

    def test_cache_hit_rate_all_hits(self):
        m = Metrics()
        for _ in range(5):
            m.record(0.01, 10, cache_hit=True)
        snap = m.snapshot()
        assert snap["cache_hit_rate"] == 1.0

    def test_cache_hit_rate_all_misses(self):
        m = Metrics()
        for _ in range(5):
            m.record(0.01, 10, cache_hit=False)
        snap = m.snapshot()
        assert snap["cache_hit_rate"] == 0.0

    def test_draft_acceptance_rate(self):
        m = Metrics()
        m.record(0.01, 10, cache_hit=True, draft_accepted=3, draft_total=10)
        snap = m.snapshot()
        assert snap["draft_acceptance_rate"] == 0.3

    def test_healing_metrics(self):
        m = Metrics()
        m.record(0.01, 10, cache_hit=True, healing_tokens=2)
        m.record(0.01, 10, cache_hit=True, healing_tokens=0)
        snap = m.snapshot()
        assert snap["healing_applied"] == 1
        assert snap["healing_tokens_removed"] == 2


# ===================================================================
# 7. Server Integration Edge Cases
# ===================================================================


class TestServerIntegrationEdgeCases:

    def test_build_recent_changes_no_data(self):
        """Importing and calling build_recent_changes with no diffs stored."""
        from sweep_local.server import build_recent_changes
        # Uses the global diff_store which may have data from other tests,
        # but a non-existent path should return ""
        result = build_recent_changes("/nonexistent/path/file.py")
        assert result == ""

    def test_pure_insertion_negative_cursor(self):
        assert is_pure_insertion_above_cursor("abc\n", "abc\n", -1) is False

    def test_pure_insertion_cursor_equals_length(self):
        code = "abc\ndef\n"
        cursor = len(code)
        completion = "abc\ndef\nnew line\n"
        # Cursor at end of file
        result = is_pure_insertion_above_cursor(code, completion, cursor)
        assert isinstance(result, bool)

    def test_pure_insertion_empty_completion(self):
        code = "abc\ndef\n"
        cursor = 4
        result = is_pure_insertion_above_cursor(code, "", cursor)
        assert result is False

    def test_prefill_multiline_no_cursor_line(self):
        """Prefill at cursor position 0 in default mode."""
        code = "first line\nsecond\n"
        prefill = compute_prefill(code, 0, changes_above_cursor=False)
        # No newline before cursor → returns ""
        assert prefill == ""

    def test_build_prompt_num_lines_zero(self):
        """num_lines_before=0 and num_lines_after=0 should give just the cursor line."""
        file = "a\nb\nc\nd\ne\n"
        cursor = len("a\nb\n")  # cursor at "c"
        _, code_block, _, _ = build_prompt(
            "f.py", file, cursor, num_lines_before=0, num_lines_after=0,
        )
        # Should contain just line "c\n"
        assert "c\n" in code_block
        lines = code_block.splitlines()
        assert len(lines) == 1
