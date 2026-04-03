# KV Cache Reuse

KV cache reuse is the biggest latency optimization in newsweep. It avoids reprocessing the prompt from scratch when consecutive requests share a common prefix, which they almost always do.

## Background: what is the KV cache?

Transformer models process input tokens through multiple attention layers. Each layer computes **key** and **value** tensors for every token. These tensors are cached so they don't need to be recomputed when generating subsequent tokens.

Without caching, generating N tokens would require processing the entire prompt N times. With caching, the prompt is processed once, and each new token only requires a single-token forward pass.

## The opportunity

In autocomplete, consecutive requests from the same file share most of their prompt text. The user types a few characters, and the prompt changes only at the end:

```
Request 1: "def compute_average(values):\n    total = sum(val"
Request 2: "def compute_average(values):\n    total = sum(values"
                                                          ^^^^
                                                    only this changed
```

The prompts share a long common prefix. If we saved the KV cache from request 1, we could reuse it for request 2 and only process the new tokens.

## How it works

newsweep maintains two global variables:

- `_last_tokens`: The token IDs from the previous request's prompt
- `_last_cache`: The KV cache state from the previous request

On each new request:

### 1. Find the common prefix

```python
def _common_prefix_len(a, b):
    """How many tokens match from the start."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))
```

### 2. Restore and trim the cache

If there's a common prefix:
1. Restore fresh `KVCache` objects from the saved snapshot (only the used portion is stored)
2. Trim to the common prefix length
3. Only feed the *new* tokens (after the common prefix) to the model

```
Saved snapshot: [(keys_slice, values_slice, 500), ...] per layer
Common prefix: 480 tokens
Restored cache: covering 480 tokens
New tokens to process: 20 (instead of 500)
```

### 3. Save cache for next request

After generation completes (and assuming the request wasn't cancelled or early-stopped), a lightweight snapshot of the cache is saved via `_snapshot_cache()`. This stores only the used portion of each layer's keys/values (not the full pre-allocated buffer), keeping memory usage tight.

The cache is **not** saved after:
- **Cancellation**: The generation was incomplete, so the cache state is unreliable
- **Early stop**: The suffix tokens appended weren't model-generated, so the cache would contain incorrect entries

## Snapshot-based caching

Instead of `copy.deepcopy` (which copies the full pre-allocated buffer including unused slots), newsweep uses a snapshot approach:

1. **At save time** (`_snapshot_cache`): Slice only the used portion (`keys[..., :offset, :]`) of each layer and force evaluation via `mx.eval`. This produces compact, contiguous arrays.
2. **At restore time** (`_restore_cache`): Create fresh `KVCache` objects and initialize them with `update_and_fetch` from the saved slices.

This avoids copying unused pre-allocated buffer space (KVCache allocates in 256-token steps) and eliminates Python's `deepcopy` overhead on complex nested objects.

## Typical hit rates

In practice, KV cache reuse achieves **60-80% hit rates** for consecutive edits in the same file. Hit rates are highest when:
- The user is editing within a small region (the broad context doesn't change)
- Keystrokes are rapid (the prompt changes by only a few tokens)
- The file structure is stable (no large-scale refactoring)

Cache misses occur when:
- The user switches to a different file (completely different prompt)
- Major edits shift the code block window significantly
- Token healing changes the encoding of shared prefix tokens

## Thread safety

`_last_tokens` and `_last_cache` are protected by `_inference_lock`. Since only one generation runs at a time, there are no races on the cache state. The lock is acquired at the start of `generate()` and held until generation completes.

## Monitoring

The `/v1/stats` endpoint reports:
- `cache_hit_rate`: Fraction of requests that reused the cache
- These are rolling metrics over the last 200 requests
