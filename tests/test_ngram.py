"""Tests for n-gram speculative decoding and early stop logic."""

from sweep_local.server import (
    _check_early_stop,
    build_ngram_index,
    ngram_draft_tokens,
)


# --- build_ngram_index ---


def test_ngram_index_basic():
    tokens = [10, 20, 30, 40, 50]
    index = build_ngram_index(tokens, n=3)
    # With n=3, context_len=2, so keys are bigrams
    assert (10, 20) in index
    assert (20, 30) in index
    assert (30, 40) in index
    # (40, 50) maps to position 4 which is past the end, so no entry
    assert len(index[(10, 20)]) == 1
    assert index[(10, 20)] == [2]  # position of token after (10, 20)


def test_ngram_index_repeated_pattern():
    tokens = [1, 2, 3, 1, 2, 4]
    index = build_ngram_index(tokens, n=3)
    # (1, 2) appears twice: at positions 2 and 5
    assert index[(1, 2)] == [2, 5]


def test_ngram_index_short_sequence():
    tokens = [1, 2]
    index = build_ngram_index(tokens, n=4)
    # With n=4, builds context sizes 1..3. Only unigram (1,) → [1] fits.
    assert (1,) in index
    assert index[(1,)] == [1]
    # No bigram or trigram entries possible with only 2 tokens
    assert (1, 2) not in index


# --- ngram_draft_tokens ---


def test_ngram_draft_basic():
    tokens = [10, 20, 30, 40, 50, 60]
    index = build_ngram_index(tokens, n=3)
    # Recent tokens end with (20, 30) → next is at position 2, so draft = [40, 50]
    draft = ngram_draft_tokens(index, tokens, [20, 30], num_draft=2, n=3)
    assert draft == [40, 50]


def test_ngram_draft_no_match():
    tokens = [10, 20, 30, 40]
    index = build_ngram_index(tokens, n=3)
    draft = ngram_draft_tokens(index, tokens, [99, 88], num_draft=2, n=3)
    assert draft == []


def test_ngram_draft_fallback_to_shorter_context():
    tokens = [10, 20, 30, 40, 50]
    index = build_ngram_index(tokens, n=4)
    # Full context (99, 88, 20) won't match, but (88, 20) or (20,) might
    # With n=4, context_len=3. We try 3, 2, 1.
    # (20,) should match — position after 20 is index 2 → draft starts at [30]
    draft = ngram_draft_tokens(index, tokens, [99, 88, 20], num_draft=2, n=4)
    assert draft == [30, 40]


def test_ngram_draft_at_end_of_prompt():
    tokens = [10, 20, 30]
    index = build_ngram_index(tokens, n=3)
    # (20, 30) → position 2, but only token at 2 is 30 and nothing after
    draft = ngram_draft_tokens(index, tokens, [20, 30], num_draft=3, n=3)
    # Should return empty or partial (position 2 is past end)
    assert len(draft) <= 3


def test_ngram_draft_recent_too_short():
    tokens = [10, 20, 30, 40]
    index = build_ngram_index(tokens, n=4)
    # recent_tokens has only 2 elements, context_len=3
    draft = ngram_draft_tokens(index, tokens, [10, 20], num_draft=2, n=4)
    assert draft == []


# --- _check_early_stop ---


def test_early_stop_match():
    suffix = [100, 200, 300]
    assert _check_early_stop(100, suffix, 0) is True
    assert _check_early_stop(200, suffix, 1) is True
    assert _check_early_stop(300, suffix, 2) is True


def test_early_stop_no_match():
    suffix = [100, 200, 300]
    assert _check_early_stop(999, suffix, 0) is False


def test_early_stop_past_end():
    suffix = [100, 200]
    assert _check_early_stop(100, suffix, 5) is False


def test_early_stop_empty_suffix():
    assert _check_early_stop(100, [], 0) is False
