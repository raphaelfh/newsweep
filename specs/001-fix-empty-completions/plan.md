# Implementation Plan: Fix Empty Completions

**Branch**: `001-fix-empty-completions` | **Date**: 2026-04-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-fix-empty-completions/spec.md`

## Summary

The server returns empty completions for all 200 benchmark scenarios. Three root causes identified: (1) prompt formatting causes the model to predict a stop token as its very first token, (2) silent exception catching masks generation errors, (3) the `is_pure_insertion_above_cursor` filter may incorrectly blank valid completions. The fix addresses all three: correct prompt construction so the model generates meaningful code tokens, add error observability, and validate post-generation filters.

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: mlx-lm >=0.31.0, FastAPI >=0.110.0, uvicorn >=0.27.0, watchdog >=4.0.0
**Storage**: N/A (in-memory KV cache, file-based model weights)
**Testing**: pytest >=8.0 (unit tests), bench_quality.py (200 scenarios), bench_latency.py
**Target Platform**: macOS Apple Silicon (M1/M2/M3/M4)
**Project Type**: Local LLM inference server (Tabby-compatible API)
**Performance Goals**: p50 < 400ms, p95 < 1500ms
**Constraints**: Temperature 0.0 (greedy decoding), MLX 4-bit quantized model, single-user local server
**Scale/Scope**: Single concurrent user, 200 benchmark scenarios, ~1,800 LOC in sweep_local/

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

No project constitution defined (template placeholders only). No gates to enforce. Proceeding.

## Project Structure

### Documentation (this feature)

```text
specs/001-fix-empty-completions/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
sweep_local/
├── __init__.py
├── server.py            # 1,168 lines — FastAPI server, generation loop, stop token handling
├── config.py            # 42 lines — configuration constants (cycle detection, repetition penalty)
├── sweep_prompt.py      # 220 lines — prompt construction, STOP_TOKENS, is_pure_insertion_above_cursor
├── file_watcher.py      # 126 lines — file change detection
└── token_healing.py     # 252 lines — partial-word tokenization fixing

tests/
├── test_behavior.py     # Core behavior tests
├── test_edge_cases.py   # Edge case handling
├── test_sweep_prompt.py # Prompt construction tests
├── bench_quality.py     # Quality benchmark (200 scenarios)
├── bench_latency.py     # Latency benchmark
└── fixtures/            # Test fixtures & scenario data
```

**Structure Decision**: Existing structure is correct for this bug fix. Changes touch `sweep_local/server.py`, `sweep_local/sweep_prompt.py`, and `sweep_local/config.py`. No new files needed in the source tree.

## Complexity Tracking

No constitution violations — section not applicable.
