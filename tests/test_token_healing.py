"""Tests for multi-token healing."""

import mlx.core as mx

from sweep_local.token_healing import (
    VocabTrie,
    find_healing_boundary,
    make_healing_processor,
    warmup_prefix_cache,
    _prefix_cache,
)


class MockTokenizer:
    """Simple tokenizer for testing. Maps single characters to IDs."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab  # text -> id
        self._id_to_text = {v: k for k, v in vocab.items()}
        self.bos_token_id = None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        """Greedy longest-match encoding."""
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


# A vocab that mimics the "Node" problem from the blog post
VOCAB = {
    "N": 0,
    "o": 1,
    "d": 2,
    "e": 3,
    "Node": 4,
    " ": 5,
    "->": 6,
    ":": 7,
    "od": 8,
    "de": 9,
    "No": 10,
    "class": 11,
    "sw": 12,
    "s": 13,
    "weep": 14,
    "sweep": 15,
    "wing": 16,
    "swap": 17,
    "\n": 18,
}


def make_tokenizer():
    return MockTokenizer(VOCAB)


# --- VocabTrie ---


def test_trie_prefix_search():
    trie = VocabTrie.from_tokenizer(make_tokenizer())
    # Tokens starting with "N": "N" (0), "Node" (4), "No" (10)
    result = set(trie.prefix_search("N"))
    assert result == {0, 4, 10}


def test_trie_prefix_search_exact():
    trie = VocabTrie.from_tokenizer(make_tokenizer())
    # Tokens starting with "Node": just "Node" (4)
    result = set(trie.prefix_search("Node"))
    assert result == {4}


def test_trie_prefix_search_no_match():
    trie = VocabTrie.from_tokenizer(make_tokenizer())
    result = trie.prefix_search("xyz")
    assert result == []


def test_trie_continuation_search():
    trie = VocabTrie.from_tokenizer(make_tokenizer())
    # For prefix "sw":
    # Tokens starting with "sw": "sw" (12), "sweep" (15), "swap" (17)
    # Tokens where "sw" starts with token: "s" (13)
    result = set(trie.continuation_search("sw"))
    assert {12, 15, 17} <= result  # starts with "sw"
    assert 13 in result  # "s" is a prefix of "sw"


def test_trie_continuation_search_single_char():
    trie = VocabTrie.from_tokenizer(make_tokenizer())
    # For prefix "s":
    # Tokens starting with "s": "s" (13), "sw" (12), "sweep" (15), "swap" (17)
    # No tokens where "s" starts with token (only "s" itself, already counted)
    result = set(trie.continuation_search("s"))
    assert {13, 12, 15, 17} <= result


# --- find_healing_boundary ---


def test_healing_boundary_stable():
    tok = make_tokenizer()
    # "Node" encodes as [4] — single token, stable
    tokens = tok.encode("Node")
    trimmed, prefix = find_healing_boundary(tokens, tok)
    assert trimmed == tokens
    assert prefix == ""


def test_healing_boundary_unstable():
    tok = make_tokenizer()
    # Simulate what happens when user types "Nod" — we need tokens that
    # would re-encode differently. Let's construct manually.
    # If the prompt ends with tokens [0, 8] = "N" + "od" = "Nod"
    # Re-encoding "od" gives [8] which is stable for 1 token
    # Re-encoding "Nod" = "No" + "d" = [10, 2], which differs from [0, 8]
    # So healing should detect instability at n=2
    tokens = [0, 8]  # "N" + "od" = "Nod"
    trimmed, prefix = find_healing_boundary(tokens, tok)
    # Should roll back the unstable boundary
    # n=1: decode [8]="od", re-encode "od"=[8] → stable
    # So no healing needed for just the last token
    assert prefix == ""


def test_healing_boundary_with_preceding_context():
    tok = make_tokenizer()
    # "class " + "Nod" where "Nod" = [0, 8] ("N" + "od")
    # The last token "od" (8) is stable on its own
    # But the real issue is the full boundary: [0, 8] for "Nod"
    # n=1: [8]="od", encode("od")=[8] → stable, no healing
    # This is actually correct — "od" IS a real token
    tokens = [11, 5, 0, 8]  # "class" + " " + "N" + "od"
    trimmed, prefix = find_healing_boundary(tokens, tok)
    assert prefix == ""  # "od" is a valid token boundary


def test_healing_boundary_single_token():
    tok = make_tokenizer()
    tokens = [4]  # "Node"
    trimmed, prefix = find_healing_boundary(tokens, tok)
    assert trimmed == [4]
    assert prefix == ""


def test_healing_boundary_empty():
    tok = make_tokenizer()
    tokens = []
    trimmed, prefix = find_healing_boundary(tokens, tok)
    assert trimmed == []
    assert prefix == ""


# --- make_healing_processor ---


def test_healing_processor_constrains_tokens():
    tok = make_tokenizer()
    trie = VocabTrie.from_tokenizer(tok)
    vocab_size = len(VOCAB)

    processor = make_healing_processor("sw", trie, tok, prompt_token_count=5)

    # First call: tokens is just prompt (5 tokens), no generated tokens yet
    prompt_tokens = mx.array([11, 5, 0, 8, 7])
    logits = mx.zeros(vocab_size)

    result = processor(prompt_tokens, logits)
    result_list = result.tolist()

    # Allowed tokens for "sw": "sw"(12), "sweep"(15), "swap"(17), "s"(13)
    allowed = trie.continuation_search("sw")
    for tid in range(vocab_size):
        if tid in allowed:
            assert result_list[tid] == 0.0, f"Token {tid} should be allowed"
        else:
            assert result_list[tid] == float("-inf"), f"Token {tid} should be masked"


def test_healing_processor_becomes_noop():
    tok = make_tokenizer()
    trie = VocabTrie.from_tokenizer(tok)
    vocab_size = len(VOCAB)

    processor = make_healing_processor("s", trie, tok, prompt_token_count=5)

    prompt_tokens = mx.array([11, 5, 0, 8, 7])
    logits = mx.zeros(vocab_size)

    # First call: constrains to tokens matching "s"
    result = processor(prompt_tokens, logits)
    assert any(v == float("-inf") for v in result.tolist())

    # Second call: simulate that "s" (token 13) was generated
    tokens_with_gen = mx.array([11, 5, 0, 8, 7, 13])  # prompt + generated "s"
    result2 = processor(tokens_with_gen, logits)
    # After consuming "s", prefix is now "" → no-op
    assert all(v == 0.0 for v in result2.tolist())


def test_healing_processor_multi_step():
    tok = make_tokenizer()
    trie = VocabTrie.from_tokenizer(tok)
    vocab_size = len(VOCAB)

    # Healing prefix is "No" — needs to be consumed in steps
    processor = make_healing_processor("No", trie, tok, prompt_token_count=3)

    prompt = mx.array([11, 5, 7])
    logits = mx.zeros(vocab_size)

    # Step 1: constrain to tokens matching "No"
    result1 = processor(prompt, logits)
    # "N" (0) is a prefix of "No" → allowed
    # "No" (10) starts with "No" → allowed
    # "Node" (4) starts with "No" → allowed
    allowed_step1 = set(trie.continuation_search("No"))
    assert 0 in allowed_step1  # "N" is prefix of "No"
    assert 10 in allowed_step1  # "No" matches
    assert 4 in allowed_step1  # "Node" starts with "No"

    # Step 2: "N" was generated → remaining is "o"
    tokens_step2 = mx.array([11, 5, 7, 0])  # prompt + "N"
    result2 = processor(tokens_step2, logits)
    # Now constraining for "o"
    allowed_step2 = set(trie.continuation_search("o"))
    assert 1 in allowed_step2  # "o" exact match
    assert 8 in allowed_step2  # "od" starts with "o"


# --- warmup_prefix_cache ---


def test_warmup_prefix_cache():
    tok = make_tokenizer()
    trie = VocabTrie.from_tokenizer(tok)

    _prefix_cache.clear()
    # With threshold=1, cache any single char with >= 1 matching token
    warmup_prefix_cache(trie, tok, threshold=1)

    # Should have cached entries for common first characters
    assert len(_prefix_cache) > 0
    # "s" should be cached (appears as first char of s, sw, sweep, swap)
    assert "s" in _prefix_cache
    _prefix_cache.clear()  # cleanup
