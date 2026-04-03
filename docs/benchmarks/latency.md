# Latency Benchmark

Measures raw speed of individual components and the full HTTP pipeline.

## Usage

```bash
# Component benchmarks (no model/server required)
python tests/bench_latency.py --mode=components --iterations=100

# HTTP benchmarks (server must be running)
python tests/bench_latency.py --mode=http --host=localhost --port=8741

# Both
python tests/bench_latency.py --mode=all --iterations=50
```

## Component benchmarks

Tests individual functions in isolation:

| Component | What it measures |
|---|---|
| `build_prompt` | Prompt construction from a 500-line file |
| `build_ngram_index` | N-gram index creation (256–2048 tokens) |
| `ngram_draft_tokens` | Draft token lookup from n-gram index |
| `deepcopy KV cache` | Cache snapshot cost (256–1024 seq length) |
| `psi_to_chunks` | PSI definition formatting (10 definitions) |
| `build_recent_changes` | Diff context assembly (5 files) |
| `find_healing_boundary` | Token healing boundary detection |

## HTTP benchmarks

End-to-end latency through the server:

| Scenario | What it measures |
|---|---|
| Cold start | Cache miss — different file between requests |
| Warm cache | Same file + cursor — maximum cache reuse |
| Cursor move | Same file, cursor shifted ±50 chars |
| Simple prompt | Raw prompt mode, no code intelligence |

## Output

Reports p50, p95, p99, and mean latency in milliseconds for each scenario. Also prints server stats (cache hit rate, draft acceptance rate, token healing count).
