# Research: Fix Empty Completions

**Feature**: 001-fix-empty-completions
**Date**: 2026-04-03

## Diagnostic Findings (2026-04-03)

**Confirmed root cause**: The `repetition_processor` in the n-gram speculative decoding path (server.py, lines 603-607) is called with 1D logits after `squeeze(0)`, but `make_repetition_penalty` from mlx_lm expects 2D logits (`logits[:, tokens]`). This throws `ValueError: Too many indices for array with 1 dimensions` on EVERY request. The `except Exception` block (server.py:823-828) catches this and returns `""`.

**Evidence**:
- Direct model calls (`generate_step`) produce correct completions for all tested scenarios
- The n-gram path calls processors manually and squeezes logits to 1D before passing to `repetition_processor`
- The standard/draft paths pass processors to `generate_step()` which handles shapes correctly
- Token healing is not involved (no rollback for tested prompts)
- The model's first predicted token is whitespace (indentation), NOT a stop token

**Initial hypotheses updated**:
- RC-1 (stop token on first sample): DISPROVEN — model predicts valid tokens
- RC-2 (silent exception): CONFIRMED — this is the mechanism producing empty strings
- RC-3 (is_pure_insertion_above_cursor): NOT REACHED — exception fires before filter runs

## Root Cause Analysis

### RC-1: Stop Token on First Sample (DISPROVEN)

**Decision**: The prompt format in `sweep_prompt.py` conditions the model to predict `<|file_sep|>` or `<|endoftext|>` as the most likely first token. The generation loop in `server.py` checks `if tid in stop_ids: break` BEFORE appending to `generated`, so the first stop token yields an empty token list and an empty completion string. With temperature=0.0 (greedy), this is deterministic — it happens every time.

**Rationale**: The prompt template ends with a `prefill` section that contains unchanged lines the model should reproduce. If the prefill already covers the expected output, or if the template's section markers confuse the model about what comes next, the model predicts "end of section" (`<|file_sep|>`) as the most probable continuation.

**Alternatives considered**:
- Masking stop tokens from logits on the first sample: Rejected — this is a workaround, not a fix. If the prompt format is wrong, the model will produce garbage even without stop tokens.
- Raising temperature: Rejected — greedy decoding (temp=0.0) is correct for autocomplete. The prompt, not the sampling, needs fixing.

### RC-2: Silent Exception Handling

**Decision**: The `except Exception` block at `server.py:823-828` catches ALL exceptions during MLX generation, logs them, and returns `""`. While it already logs via `logger.exception()`, the benchmark and client see no difference between "model returned nothing" and "model crashed." The fix should ensure error logging is always enabled and visible during benchmark runs.

**Rationale**: Exceptions may mask secondary issues (MLX runtime errors, shape mismatches, OOM). Even if RC-1 is the primary cause, silenced errors would delay diagnosis of other problems.

**Alternatives considered**:
- Returning error responses to client: Rejected — breaks Tabby API contract and could crash IDE plugin.
- Removing the catch entirely: Rejected — server must stay up even if individual requests fail.

### RC-3: `is_pure_insertion_above_cursor` Filter

**Decision**: The filter at `sweep_prompt.py:83-107` blanks completions that only insert above the cursor without editing the cursor line. This is a valid heuristic for IDE UX, but it may be triggered incorrectly if the prompt's `relative_cursor` value is miscalculated or if the `code_block` boundaries are wrong.

**Rationale**: The filter checks `completion.startswith(prefix) and completion.endswith(cursor_line + suffix)`. If the prompt construction sets `relative_cursor` incorrectly (e.g., off-by-one), valid completions get blanked.

**Alternatives considered**:
- Removing the filter: Rejected — it serves a valid UX purpose (preventing confusing insertions above where the developer is typing).
- Adding a logging path: Chosen — log when this filter triggers so we can measure its false-positive rate.

## Fix Strategy

### Step 1: Diagnose prompt format (server logs)

Add temporary diagnostic logging to capture:
- The full prompt text sent to the model (first 500 chars + last 200 chars)
- The first token ID sampled and its decoded value
- Whether generation ended via stop token, cycle detection, max tokens, or exception

Run the benchmark with diagnostics enabled. This confirms which root cause is active.

### Step 2: Fix prompt construction

Based on diagnostic output, fix `build_prompt()` in `sweep_prompt.py` so that:
- The prefill section correctly positions the model to generate the NEXT tokens (not to repeat what's already there)
- The `<|file_sep|>` markers are correctly placed so the model doesn't interpret them as "end of output"
- The cursor position and code block boundaries are correct

### Step 3: Harden post-generation pipeline

- Ensure cycle trimming never produces an empty result (preserve at least the first non-cyclic tokens)
- Add logging when `is_pure_insertion_above_cursor` triggers
- Verify `relative_cursor` calculation is correct

### Step 4: Validate

1. Run unit tests: `python -m pytest tests/ -x`
2. Run quality benchmark: `python tests/bench_quality.py`
3. Verify: exact match > 10%, LCP ratio > 30%, zero empty completions
4. Run latency benchmark: `python tests/bench_latency.py --mode=components`
5. Verify: p50 < 400ms, p95 < 1500ms

## Key Files

| File | Lines | Role |
|------|-------|------|
| `sweep_local/sweep_prompt.py:8-16` | Prompt template | `PROMPT_TEMPLATE` with `<|file_sep|>` markers |
| `sweep_local/sweep_prompt.py:24-34` | Stop tokens | `STOP_TOKENS` list definition |
| `sweep_local/sweep_prompt.py:83-107` | Post-gen filter | `is_pure_insertion_above_cursor()` |
| `sweep_local/sweep_prompt.py:110-202` | Prompt builder | `build_prompt()` function |
| `sweep_local/server.py:233-240` | Stop ID init | `stop_ids` set built from `STOP_TOKENS` |
| `sweep_local/server.py:457-828` | Generation loop | All three generation paths (n-gram, draft, standard) |
| `sweep_local/server.py:609-617` | First token check | Stop check BEFORE append in n-gram path |
| `sweep_local/server.py:823-828` | Exception handler | Catch-all returns `""` |
| `sweep_local/server.py:834-839` | Cycle trimming | Removes repeated token cycles |
| `sweep_local/server.py:906-921` | Cycle detection | `_has_token_cycle()` function |
| `sweep_local/config.py:23-26` | Cycle config | `CYCLE_DETECT_WINDOW`, `REPETITION_PENALTY` |
