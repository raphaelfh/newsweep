# CLAUDE.md

Fast local autocomplete server for JetBrains IDEs. Runs sweep-next-edit-v2-7B (MLX 4-bit) on Apple Silicon via a Tabby-compatible API (port 8741).

## Priority

Low latency and high quality completions. Every decision should favor speed and accuracy on M-series chips.

## Rules

- Keep docs (`/docs`) in sync with code.
- Write everything in English.

## Validation workflow

After any change to the inference pipeline, prompt construction, or server logic (`sweep_local/`):

1. Run unit tests: `python -m pytest tests/ -x`
2. Run quality benchmark against the live server: `python tests/bench_quality.py --host=localhost --port=8741 --output=results.json`
3. Compare results against the previous run — check for regressions in exact match rate, prefix match rate, and LCP ratio.
4. Run latency benchmark: `python tests/bench_latency.py --mode=components`

The quality benchmark uses 100 realistic editing scenarios in `tests/fixtures/scenarios.json` covering line completion, block completion, refactoring, bug fixes, and structural edits. It measures how well the model predicts what a developer would type next.

Never skip step 2 — latency improvements that degrade prediction quality are regressions.

## Active Technologies
- Python 3.12+ + mlx-lm >=0.31.0, FastAPI >=0.110.0, uvicorn >=0.27.0, watchdog >=4.0.0 (001-fix-empty-completions)
- N/A (in-memory KV cache, file-based model weights) (001-fix-empty-completions)

## Recent Changes
- 001-fix-empty-completions: Added Python 3.12+ + mlx-lm >=0.31.0, FastAPI >=0.110.0, uvicorn >=0.27.0, watchdog >=4.0.0
