# Inference Pipeline

This document traces a completion request through the entire inference pipeline in `server.py`, from prompt encoding to decoded output.

## Pipeline stages

```
prompt string
    |
    v
[1] Tokenize
    |
    v
[2] Token healing + repetition penalty (logits processors)
    |
    v
[3] KV cache reuse (find common prefix with previous request)
    |
    v
[4] Generation loop (one of three paths)
    |-- N-gram speculation (default)
    |-- Draft-model speculation
    |-- Standard generation
    |
    v
[5] Post-processing (strip healing prefix, trim at stop tokens)
    |
    v
completion string
```

## Stage 1: Tokenize

The prompt string is encoded to token IDs using the model's tokenizer. The resulting token list is used for cache comparison and fed to the model.

## Stage 2: Token healing

If enabled, the last few tokens are checked for **unstable tokenization boundaries**. This happens when partial words (like "Nod" for "Node") produce different token sequences than the full word would.

See [Token Healing](../deep-dives/token-healing.md) for the full algorithm. The output is:
- A potentially shorter token list (with unstable tail tokens removed)
- A healing prefix string
- A logits processor that constrains the first few generation steps

## Stage 3: KV cache reuse

The server compares the current prompt tokens with the previous request's tokens (`_last_tokens`). If they share a common prefix, the KV cache from the previous request is reused:

```
Previous:  [tok1, tok2, tok3, tok4, tok5, tok6]
Current:   [tok1, tok2, tok3, tok7, tok8]
                                ^
                          common prefix = 3 tokens

Cache is deep-copied, trimmed to 3 entries, and reused.
Only [tok7, tok8] need to be processed through the model.
```

See [KV Cache Reuse](../deep-dives/kv-cache-reuse.md) for details.

## Stage 4: Generation loop

The generation loop produces tokens one at a time, with three possible code paths:

### Path A: N-gram speculation (default)

The fastest path for code completion. Instead of using a separate draft model, it looks up patterns in the prompt tokens themselves to propose candidate tokens.

```
1. Process prompt through model -> get logits
2. Loop:
   a. Sample base token from logits
   b. Look up n-gram matches in prompt
   c. If match found:
      - Feed [base_token, draft0, draft1, ...] through model
      - Verify each draft token against model's prediction
      - Accept consecutive matches, reject at first mismatch
   d. If no match: feed base token through model
   e. Continue with updated logits
```

See [N-gram Speculation](../deep-dives/ngram-speculation.md) for the algorithm.

### Path B: Draft-model speculation

Used when n-gram speculation is disabled. A smaller model (Qwen2.5-0.5B) proposes candidate tokens that the main model verifies. Uses MLX's built-in `speculative_generate_step()`.

### Path C: Standard generation

Simple token-by-token generation. Uses MLX's `generate_step()` iterator. Slowest but simplest path.

### Checks on every token

Regardless of the generation path, every produced token is checked for:

1. **Stop tokens**: If the token is a stop token (EOS, `<|file_sep|>`, etc.), generation halts.
2. **Request cancellation**: If `_request_seq > seq`, a newer request has arrived. Generation aborts and returns empty.
3. **Early stop**: If the token matches the next expected token in the known suffix, a counter increments. After `EARLY_STOP_MATCH_TOKENS` consecutive matches, the remaining suffix is appended and generation stops. See [Early Cancellation](../deep-dives/early-cancellation.md).
4. **Cycle detection**: If the last `2*W` generated tokens form two identical copies of a `W`-length pattern (for any W from 4 to `CYCLE_DETECT_WINDOW`), generation stops and the duplicate cycle is trimmed from the output. This is a safety net against repetition loops.

## Stage 5: Post-processing

The generated token IDs are decoded back to text, then:

1. **Strip healing prefix**: If token healing was applied, the healing prefix is removed from the start of the completion (the IDE already has this text).
2. **Trim at stop tokens**: Any stop token text found in the completion is used as a cut point.

Back in the request handler, additional processing:

3. **Pure insertion rejection**: If the completion only inserts text above the cursor line without editing it, the completion is discarded (returned as empty).
4. **Cursor extraction**: The model outputs the full updated code from the prefill end. This includes the beginning of the cursor line (before the cursor position) and any suffix lines. Both are stripped so the returned completion contains only the text to be inserted at the cursor position.

## Metrics

Every generation records metrics to a rolling window (200 requests):

| Metric | Description |
|---|---|
| Latency | Wall-clock time for generation |
| Tokens/sec | Generation throughput |
| Cache hit | Whether KV cache was reused |
| Draft accepted/total | Speculative decoding acceptance rate |
| Healing applied | Whether token healing was active |
| Cancellations | Requests aborted due to newer request |

Access metrics via `GET /v1/stats`. See [API Endpoints](../api/endpoints.md).

## Concurrency

The `_inference_lock` ensures only one generation runs at a time. The request handler (a sync `def`) runs in FastAPI's thread pool, so multiple requests can queue up but only one enters `generate()`.

The cancellation mechanism means queued requests don't waste time: when a request finally acquires the lock, it first checks if it's already been superseded. If so, it returns immediately.
