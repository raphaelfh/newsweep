# Configuration

All settings live in `sweep_local/config.py`. There are no environment variables or config files -- edit the Python constants directly and restart the server.

## Server

| Setting | Default | Description |
|---|---|---|
| `HOST` | `"0.0.0.0"` | Bind address |
| `PORT` | `8741` | Server port |

## Prompt construction

These control the code block extracted around the cursor, which is the core input to the model.

| Setting | Default | Description |
|---|---|---|
| `NUM_LINES_BEFORE` | `7` | Lines above cursor included in the code block |
| `NUM_LINES_AFTER` | `7` | Lines below cursor included in the code block |
| `MAX_NEW_TOKENS` | `64` | Maximum tokens the model can generate per request |

The code block is what the model "rewrites". Larger windows give more context but increase latency. The default 7+7 balances context with speed for autocomplete.

Beyond the code block, the prompt also includes a **broad context window** of ~50 lines above and below the cursor (excluding the code block) to give the model file-level awareness. This window is snapped to 40-line chunk boundaries so small cursor movements don't invalidate the KV cache.

## Speculative decoding

Speculative decoding generates multiple tokens per forward pass by *proposing* candidate tokens and *verifying* them in batch. See [N-gram Speculation](../deep-dives/ngram-speculation.md) for how this works.

| Setting | Default | Description |
|---|---|---|
| `USE_NGRAM_SPECULATION` | `True` | Use n-gram lookup for draft proposals (recommended) |
| `NGRAM_N` | `3` | N-gram size -- matches last N-1 tokens to find proposals |
| `NUM_DRAFT_TOKENS` | `2` | Tokens proposed per speculation step |
| `DRAFT_MODEL_PATH` | `models/qwen2.5-0.5b-4bit` | Path to draft model (only used if n-gram disabled) |

**N-gram vs. draft model**: N-gram speculation is the default because it needs no extra model and is faster for code rewrites where prompts contain repetitive patterns. The draft model (Qwen2.5-0.5B) is a fallback for cases where prompt tokens don't provide good n-gram matches.

## Repetition penalty

Prevents the model from getting stuck in repetitive output loops (e.g., generating the same line over and over). Uses MLX's built-in repetition penalty logits processor.

| Setting | Default | Description |
|---|---|---|
| `REPETITION_PENALTY` | `1.2` | Penalty factor for already-generated tokens (1.0 = disabled) |
| `REPETITION_CONTEXT_SIZE` | `64` | How many recent tokens the penalty considers |
| `CYCLE_DETECT_WINDOW` | `12` | Max cycle length for the safety-net cycle detector |

The penalty reduces the probability of tokens that have already appeared in the recent context. Values above 1.0 discourage repetition; higher values are more aggressive. The cycle detector is a fallback that force-stops generation if the same token sequence repeats back-to-back, trimming the duplicate from the output.

## Token healing

Token healing fixes broken completions caused by partial-word tokenization boundaries. See [Token Healing](../deep-dives/token-healing.md).

| Setting | Default | Description |
|---|---|---|
| `ENABLE_TOKEN_HEALING` | `True` | Enable/disable token healing |
| `MAX_HEALING_TOKENS` | `3` | Max tokens to roll back when detecting unstable boundaries |

## Early cancellation

Stops generation early when the model is reproducing the known suffix (code after the cursor). See [Early Cancellation](../deep-dives/early-cancellation.md).

| Setting | Default | Description |
|---|---|---|
| `EARLY_STOP_MATCH_TOKENS` | `4` | Consecutive suffix-matching tokens needed to trigger early stop |

Lower values stop earlier (faster, but risk cutting off genuine edits). Higher values are safer but reduce the speedup.

## File watcher

The file watcher monitors your project for saves and records diffs, which are injected into prompts as "recent changes" context. See [File Watcher](../deep-dives/file-watcher.md).

| Setting | Default | Description |
|---|---|---|
| `PROJECT_ROOT` | `~/PycharmProjects` | Root directory to watch for file changes |
| `MAX_DIFFS_PER_FILE` | `10` | Rolling buffer size per file |
| `MAX_CROSS_FILE_DIFFS` | `3` | Max diffs from *other* files included in prompt |

## PSI context

PSI (Program Structure Interface) definitions are type-resolved symbols sent by JetBrains IDEs. These provide the model with signatures of classes, methods, and types relevant to the cursor position.

| Setting | Default | Description |
|---|---|---|
| `MAX_PSI_TOKENS` | `1500` | Token budget for PSI definitions in the prompt |
| `MAX_PSI_DEFINITIONS` | `10` | Max definitions per request |

When PSI definitions are present, the broad context window shrinks further (from 50 to 25 lines per side) to keep the total prompt length manageable.
