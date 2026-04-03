# Quickstart: Fix Empty Completions

**Feature**: 001-fix-empty-completions

## Prerequisites

- macOS Apple Silicon (M1+)
- Python 3.12+
- Model downloaded: `models/sweep-next-edit-v2-7B/`
- Virtual environment: `.venv/`

## Setup

```bash
# Activate venv
source .venv/bin/activate

# Start the server
python -m sweep_local.server
# Server runs on http://localhost:8741
```

## Validation Commands

### 1. Unit tests (fast, no server needed)

```bash
python -m pytest tests/ -x
```

### 2. Quality benchmark (requires running server)

```bash
python tests/bench_quality.py --host=localhost --port=8741 --output=results.json
```

**Expected after fix**:
- Exact match > 10% (currently 0%)
- LCP ratio > 30% (currently 0%)
- Zero empty completions (currently 200/200)

### 3. Latency benchmark (requires running server)

```bash
python tests/bench_latency.py --mode=components
```

**Expected after fix**:
- p50 < 400ms
- p95 < 1500ms

## Key Files to Modify

| File | What to change |
|------|----------------|
| `sweep_local/sweep_prompt.py` | Fix prompt template, validate `build_prompt()` output, add logging to `is_pure_insertion_above_cursor` |
| `sweep_local/server.py` | Add diagnostic logging for first-token stop, harden cycle trimming, improve error context |
| `sweep_local/config.py` | Adjust cycle detection parameters if needed |

## Debugging Tips

- Server logs go to `logs/` directory and stdout
- To see what the model receives, add a log line in `generate()` before the inference loop
- To see what token the model predicts first, log `tid` after the first `sampler()` call
- The benchmark's `--output` flag saves per-scenario results for detailed analysis
