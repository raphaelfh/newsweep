"""
Multi-token healing for autocomplete.

When users type partial words (e.g. "Nod" intending "Node"), tokenization
creates unusual token boundaries that the model hasn't seen in training.
Token healing rolls back to the last stable tokenization boundary and
constrains generation to match the user's typed prefix.

Based on: https://blog.sweep.dev/posts/token-healing-autocomplete
"""

from __future__ import annotations

import logging
from typing import Callable

import mlx.core as mx

logger = logging.getLogger("newsweep")


class VocabTrie:
    """Trie over tokenizer vocabulary for fast prefix lookup."""

    __slots__ = ("root",)

    def __init__(self):
        # Each node: dict mapping char -> child_node
        # Special key None -> list of token_ids stored at this node
        self.root: dict = {}

    @classmethod
    def from_tokenizer(cls, tokenizer) -> VocabTrie:
        """Build trie from tokenizer vocabulary. Called once at startup."""
        trie = cls()
        vocab = tokenizer.get_vocab()
        for text, token_id in vocab.items():
            trie._insert(text, token_id)
        return trie

    def _insert(self, text: str, token_id: int):
        node = self.root
        for ch in text:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        if None not in node:
            node[None] = []
        node[None].append(token_id)

    def _collect_all(self, node: dict) -> list[int]:
        """Collect all token IDs in the subtree rooted at node."""
        result = []
        stack = [node]
        while stack:
            n = stack.pop()
            if None in n:
                result.extend(n[None])
            for k, child in n.items():
                if k is not None:
                    stack.append(child)
        return result

    def prefix_search(self, prefix: str) -> list[int]:
        """Return all token IDs whose text starts with `prefix`."""
        node = self.root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        return self._collect_all(node)

    def continuation_search(self, prefix: str) -> list[int]:
        """Return token IDs valid for multi-token healing.

        A token is valid if:
        - Its text starts with `prefix` (extends the prefix), OR
        - `prefix` starts with the token text (consumes part of the prefix)
        """
        result = []

        # Tokens that start with prefix (extend it)
        result.extend(self.prefix_search(prefix))

        # Tokens where prefix starts with the token text (prefix is a prefix of the token text)
        # Walk down the trie along the prefix chars, collecting tokens at each intermediate node
        node = self.root
        for ch in prefix:
            if None in node:
                # Tokens ending here are a prefix of our prefix string
                result.extend(node[None])
            if ch not in node:
                break
            node = node[ch]

        return result


# Pre-computed cache for prefixes with many matches (e.g., space)
_prefix_cache: dict[str, list[int]] = {}


def warmup_prefix_cache(trie: VocabTrie, tokenizer, threshold: int = 1000):
    """Pre-compute allowed token sets for high-frequency single-char prefixes."""
    _prefix_cache.clear()
    vocab = tokenizer.get_vocab()
    # Find single characters that appear as prefixes of many tokens
    first_chars: dict[str, int] = {}
    for text in vocab:
        if text:
            ch = text[0]
            first_chars[ch] = first_chars.get(ch, 0) + 1

    for ch, count in first_chars.items():
        if count >= threshold:
            _prefix_cache[ch] = trie.continuation_search(ch)
            logger.info(
                f"Token healing: cached prefix {ch!r} ({len(_prefix_cache[ch])} tokens)"
            )


def find_healing_boundary(
    tokens: list[int], tokenizer, max_rollback: int = 3
) -> tuple[list[int], str]:
    """Find the last stable tokenization boundary.

    Walks backward from the end of the token list. Decodes the last N tokens
    to text, then re-encodes. If re-encoding differs, those tokens sit at an
    unstable boundary.

    Returns:
        (trimmed_tokens, healing_prefix): tokens to feed to the model, and the
        text removed that must be constrained during generation.
    """
    if len(tokens) <= 1:
        return tokens, ""

    for n in range(1, min(max_rollback + 1, len(tokens))):
        tail_tokens = tokens[-n:]
        tail_text = tokenizer.decode(tail_tokens)

        # Re-encode the tail text to see if tokenization is stable
        re_encoded = tokenizer.encode(tail_text)
        # Strip any BOS token the tokenizer might add
        if (
            re_encoded
            and hasattr(tokenizer, "bos_token_id")
            and tokenizer.bos_token_id is not None
            and re_encoded[0] == tokenizer.bos_token_id
        ):
            re_encoded = re_encoded[1:]

        if re_encoded == tail_tokens:
            # This boundary is stable
            if n == 1:
                # Last token is stable, no healing needed
                return tokens, ""
            else:
                # Previous iteration found an unstable boundary;
                # roll back n-1 tokens
                rollback = n - 1
                healing_text = tokenizer.decode(tokens[-rollback:])
                return tokens[:-rollback], healing_text

    # All checked tokens are unstable — roll back max_rollback
    rollback = min(max_rollback, len(tokens) - 1)
    healing_text = tokenizer.decode(tokens[-rollback:])
    return tokens[:-rollback], healing_text


def make_healing_processor(
    healing_prefix: str,
    trie: VocabTrie,
    tokenizer,
    prompt_token_count: int,
) -> Callable[[mx.array, mx.array], mx.array]:
    """Create a logits processor that constrains generation to match healing_prefix.

    The processor tracks how much of the prefix has been consumed across
    generation steps. Once fully consumed, it becomes a no-op.

    generate_step calls logits_processors with (all_tokens_so_far, logits).
    On call N, tokens[-1] is the token sampled at step N-1. On the first call,
    tokens only contains prompt tokens — nothing has been generated yet.

    Args:
        healing_prefix: The text that was removed and must be regenerated.
        trie: VocabTrie for fast token lookup.
        tokenizer: The tokenizer (for decode).
        prompt_token_count: Number of prompt tokens (to distinguish from generated).

    Returns:
        A callable (tokens, logits) -> logits suitable for generate_step's
        logits_processors parameter.
    """
    remaining = [healing_prefix]  # mutable container for closure
    vocab_size = [0]  # cached, set on first call

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        prefix = remaining[0]
        if not prefix:
            return logits  # no-op after prefix consumed

        # Update state based on previously generated tokens.
        # tokens contains prompt + generated tokens. Generated tokens start
        # at index prompt_token_count.
        n_generated = tokens.size - prompt_token_count
        if n_generated > 0:
            # A token was sampled since our last call — update remaining prefix
            last_token = tokens[-1].item()
            token_text = tokenizer.decode([last_token])
            if prefix.startswith(token_text):
                remaining[0] = prefix[len(token_text):]
                prefix = remaining[0]
            elif token_text.startswith(prefix):
                remaining[0] = ""
                return logits  # prefix fully consumed
            else:
                # Unexpected token — abandon healing to avoid stuck generation
                logger.warning(
                    f"Token healing: unexpected token {token_text!r} for prefix {prefix!r}"
                )
                remaining[0] = ""
                return logits

        if not prefix:
            return logits

        # Get allowed token IDs
        if prefix in _prefix_cache:
            allowed = _prefix_cache[prefix]
        else:
            allowed = trie.continuation_search(prefix)

        if not allowed:
            logger.warning(f"Token healing: no tokens match prefix {prefix!r}, skipping")
            remaining[0] = ""
            return logits

        # Mask disallowed tokens by setting their logits to -inf
        if not vocab_size[0]:
            vocab_size[0] = logits.shape[-1]
        mask = mx.full((vocab_size[0],), float("-inf"))
        allowed_array = mx.array(allowed)
        mask[allowed_array] = 0.0
        logits = logits + mask

        return logits

    return processor


