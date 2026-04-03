# Architecture Overview

## System diagram

```
 JetBrains IDE (Tabby plugin)
        |
        |  POST /v1/completions
        v
 +----------------------------------------------+
 |  newsweep server (FastAPI)                    |
 |                                               |
 |  server.py                                    |
 |  +------------------------------------------+ |
 |  | 1. Request handling                      | |
 |  |    - Parse Tabby request                 | |
 |  |    - Increment _request_seq              | |
 |  |    - Acquire _inference_lock             | |
 |  +------------------------------------------+ |
 |  | 2. Prompt construction                   | |
 |  |    - build_recent_changes() [diffs]      | |
 |  |    - psi_to_chunks() [definitions]       | |
 |  |    - build_prompt() [sweep template]     | |
 |  +------------------------------------------+ |
 |  | 3. Inference                             | |
 |  |    - Token healing (boundary fix)        | |
 |  |    - KV cache reuse (prefix sharing)     | |
 |  |    - Generation loop (with speculation)  | |
 |  |    - Early stop (suffix matching)        | |
 |  +------------------------------------------+ |
 |  | 4. Post-processing                       | |
 |  |    - Strip healing prefix                | |
 |  |    - Trim at stop tokens                 | |
 |  |    - Reject pure insertions above cursor | |
 |  +------------------------------------------+ |
 |                                               |
 |  file_watcher.py                              |
 |  +------------------------------------------+ |
 |  | watchdog Observer (daemon thread)         | |
 |  | Monitors PROJECT_ROOT for file saves      | |
 |  | Feeds diffs into global diff_store        | |
 |  +------------------------------------------+ |
 +----------------------------------------------+
        |
        |  MLX framework
        v
 +----------------------------------------------+
 |  Apple Silicon GPU (unified memory)           |
 |  - sweep-next-edit-v2-7B (main model, 4-bit) |
 |  - Qwen2.5-0.5B (draft model, optional)      |
 +----------------------------------------------+
```

## Modules

All source code lives in `sweep_local/`:

| Module | Responsibility |
|---|---|
| `server.py` | FastAPI app, inference loop, KV cache, speculation, metrics |
| `sweep_prompt.py` | Prompt template, code block extraction, prefill computation |
| `file_watcher.py` | Watchdog-based file monitor, diff store |
| `config.py` | All tunable constants |
| `token_healing.py` | Vocabulary trie, boundary detection, logits processor |

## Concurrency model

MLX runs on a single Apple Silicon GPU. This shapes the entire concurrency design:

1. **`_inference_lock`** (threading.Lock): Only one generation runs at a time. This also protects the KV cache state (`_last_tokens` / `_last_cache`) from concurrent access.

2. **Request cancellation** (`_request_seq`): When a new request arrives, it increments a global counter. The in-flight generation loop checks this counter on every token -- if a newer request exists, it stops immediately and returns an empty completion. This keeps the server responsive during fast typing.

3. **Sync endpoint**: The `/v1/completions` handler is a regular `def` (not `async def`). FastAPI runs it in a thread pool, keeping the async event loop free for health checks, stats, and other lightweight endpoints.

4. **File watcher**: Runs as a daemon thread via watchdog's `Observer`. It modifies `diff_store` which is protected by its own internal lock.

## Request lifecycle

1. IDE sends `POST /v1/completions` with code context
2. Server increments `_request_seq` and records the current value as `seq`
3. Server builds the prompt:
   - Gathers recent diffs from `diff_store` (same-file + cross-file)
   - Converts PSI definitions to retrieval chunks (if provided)
   - Calls `build_prompt()` to assemble the sweep model template
   - Seeds the diff store with the current file content
4. Server calls `generate()`:
   - Encodes prompt to tokens
   - Applies token healing if needed (rolls back unstable boundary, creates logits processor)
   - Acquires `_inference_lock`
   - Reuses KV cache from previous request if prompts share a prefix
   - Runs generation (n-gram speculation, draft model, or standard)
   - Checks for early stop on each token
   - Checks for cancellation on each token
   - Saves cache state for next request
5. Server post-processes:
   - Strips healing prefix (IDE already has this text)
   - Trims at stop tokens
   - Rejects completions that only insert above the cursor
6. Returns `TabbyCompletionResponse` with the completion text

## Data flow

```
File saves ──> file_watcher.py ──> diff_store (rolling buffer)
                                        |
IDE request ──> server.py ──────────────+──> build_prompt()
                    |                           |
                    |                    prompt string
                    |                           |
                    +──> generate() ────────────+
                    |        |
                    |   MLX inference
                    |        |
                    +──> completion text ──> IDE
```

The diff store is the bridge between the background file watcher and the request handler. When the IDE sends a completion request, the server pulls recent diffs for the current file (and neighboring files) to inject as "recent changes" context.
