"""
Latency benchmarks for the newsweep completion pipeline.

Two modes:
  --mode=components   Benchmark individual components (no model required)
  --mode=http         Benchmark full pipeline via HTTP (server must be running)

Usage:
  python tests/bench_latency.py --mode=components
  python tests/bench_latency.py --mode=http --host=localhost --port=8741
  python tests/bench_latency.py --mode=all --iterations=50
"""

import argparse
import copy
import statistics
import sys
import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def report(name: str, times_s: list[float]):
    times_ms = [t * 1000 for t in times_s]
    print(
        f"  {name:<35s}  "
        f"p50={percentile(times_ms, 50):7.2f}ms  "
        f"p95={percentile(times_ms, 95):7.2f}ms  "
        f"p99={percentile(times_ms, 99):7.2f}ms  "
        f"mean={statistics.mean(times_ms):7.2f}ms"
    )


def bench(fn, iterations: int) -> list[float]:
    """Run fn() iterations times, return list of elapsed seconds."""
    # Warmup
    for _ in range(min(3, iterations)):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Realistic test data
# ---------------------------------------------------------------------------

REALISTIC_FILE = "\n".join(
    [
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "",
        "class Config:",
        '    host: str = "localhost"',
        "    port: int = 8080",
        "    debug: bool = False",
        "",
        "    def validate(self):",
        "        if self.port < 0 or self.port > 65535:",
        '            raise ValueError(f"Invalid port: {self.port}")',
        "",
    ]
    + [f"    def method_{i}(self):\n        return {i}\n" for i in range(200)]
    + [
        "",
        "class Server:",
        "    def __init__(self, config: Config):",
        "        self.config = config",
        "        self._running = False",
        "",
        "    def start(self):",
        "        self.config.validate()",
        "        self._running = True",
        '        print(f"Listening on {self.config.host}:{self.config.port}")',
        "",
        "    def stop(self):",
        "        self._running = False",
        "",
        '    def handle_request(self, path: str, method: str = "GET"):',
        '        if method == "GET":',
        "            return self._do_get(path)",
        '        elif method == "POST":',
        "            return self._do_post(path)",
        "",
    ]
)


def make_mock_kv_cache(num_layers: int = 32, n_kv_heads: int = 4,
                       seq_len: int = 1024, head_dim: int = 128):
    """Create a realistic mock KV cache matching a 7B model layout."""
    from mlx_lm.models.cache import KVCache

    cache = []
    for _ in range(num_layers):
        c = KVCache()
        # Simulate having processed seq_len tokens
        keys = mx.zeros((1, n_kv_heads, seq_len, head_dim), dtype=mx.float16)
        values = mx.zeros((1, n_kv_heads, seq_len, head_dim), dtype=mx.float16)
        c.update_and_fetch(keys, values)
        cache.append(c)
    mx.eval(*[c.keys for c in cache], *[c.values for c in cache])
    return cache


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------

def bench_components(iterations: int):
    print(f"\n{'=' * 70}")
    print(f"Component Benchmarks ({iterations} iterations each)")
    print(f"{'=' * 70}\n")

    # --- build_prompt ---
    from sweep_local.sweep_prompt import build_prompt

    cursor_pos = len(REALISTIC_FILE) // 2
    times = bench(
        lambda: build_prompt(
            file_path="test/config.py",
            file_contents=REALISTIC_FILE,
            cursor_position=cursor_pos,
            recent_changes="",
            num_lines_before=10,
            num_lines_after=10,
        ),
        iterations,
    )
    report("build_prompt (500-line file)", times)

    # --- build_ngram_index ---
    from sweep_local.server import build_ngram_index

    for size in [256, 512, 1024, 2048]:
        tokens = list(range(size))
        times = bench(lambda t=tokens: build_ngram_index(t), iterations)
        report(f"build_ngram_index ({size} tokens)", times)

    # --- ngram_draft_tokens ---
    from sweep_local.server import ngram_draft_tokens

    tokens_1k = list(range(1024))
    idx = build_ngram_index(tokens_1k)
    times = bench(
        lambda: ngram_draft_tokens(idx, tokens_1k, tokens_1k[-3:], 3),
        iterations,
    )
    report("ngram_draft_tokens (1024 tokens)", times)

    # --- copy.deepcopy on KV cache ---
    for seq_len in [256, 512, 1024]:
        cache = make_mock_kv_cache(seq_len=seq_len)
        times = bench(lambda c=cache: copy.deepcopy(c), iterations)
        report(f"deepcopy KV cache ({seq_len} seq)", times)

    # --- psi_to_chunks ---
    from sweep_local.server import PsiDefinition, psi_to_chunks

    psi_defs = [
        PsiDefinition(
            kind="class",
            name=f"Class{i}",
            qualified_name=f"com.example.Class{i}",
            signature=f"class Class{i}(Base{i}):",
            body_preview=f"    def method(self):\n        return {i}",
            priority=10 - i,
        )
        for i in range(10)
    ]
    times = bench(lambda: psi_to_chunks(psi_defs), iterations)
    report("psi_to_chunks (10 definitions)", times)

    # --- build_recent_changes ---
    from sweep_local.file_watcher import DiffStore

    store = DiffStore()
    for i in range(5):
        fp = f"file_{i}.py"
        store.seed_file(fp, f"original content {i}\nline2\nline3")
        store.update_file(fp, f"modified content {i}\nline2\nline3\nnew line")

    # Monkey-patch the global diff_store temporarily
    import sweep_local.server as srv
    from sweep_local.server import build_recent_changes
    original_store = srv.diff_store
    srv.diff_store = store
    try:
        times = bench(lambda: build_recent_changes("file_0.py"), iterations)
        report("build_recent_changes (5 files)", times)
    finally:
        srv.diff_store = original_store

    # --- Token healing (with mock tokenizer) ---
    from sweep_local.token_healing import VocabTrie, find_healing_boundary, warmup_prefix_cache

    class MockTokenizer:
        def __init__(self):
            self._vocab = {f"tok_{i}": i for i in range(100)}
            self._vocab.update({
                "def": 100, " ": 101, "f": 102, "(": 103, ")": 104,
                ":": 105, "\n": 106, "  ": 107, "return": 108,
                "self": 109, ".": 110, "Nod": 111, "Node": 112,
                "N": 113, "od": 114, "e": 115, "No": 116, "de": 117,
            })
            self._id_to_tok = {v: k for k, v in self._vocab.items()}

        def get_vocab(self):
            return dict(self._vocab)

        def encode(self, text, add_special_tokens=False):
            tokens = []
            i = 0
            while i < len(text):
                best = None
                for length in range(min(10, len(text) - i), 0, -1):
                    chunk = text[i:i + length]
                    if chunk in self._vocab:
                        best = (chunk, self._vocab[chunk])
                        break
                if best:
                    tokens.append(best[1])
                    i += len(best[0])
                else:
                    i += 1
            return tokens

        def decode(self, ids, skip_special_tokens=False):
            return "".join(self._id_to_tok.get(i, "?") for i in ids)

        def convert_ids_to_tokens(self, ids):
            return [self._id_to_tok.get(i, "?") for i in ids]

    mock_tok = MockTokenizer()
    trie = VocabTrie.from_tokenizer(mock_tok)
    warmup_prefix_cache(trie, mock_tok)

    test_tokens = mock_tok.encode("def self.Nod")
    times = bench(
        lambda: find_healing_boundary(list(test_tokens), mock_tok, max_rollback=3),
        iterations,
    )
    report("find_healing_boundary", times)


# ---------------------------------------------------------------------------
# HTTP benchmarks
# ---------------------------------------------------------------------------

def bench_http(iterations: int, host: str, port: int):
    try:
        import httpx
    except ImportError:
        print("httpx required for HTTP benchmarks: pip install httpx")
        sys.exit(1)

    base_url = f"http://{host}:{port}"
    print(f"\n{'=' * 70}")
    print(f"HTTP Benchmarks → {base_url} ({iterations} iterations each)")
    print(f"{'=' * 70}\n")

    # Check server is running
    client = httpx.Client(base_url=base_url, timeout=30.0)
    try:
        r = client.get("/v1/health")
        r.raise_for_status()
        print(f"  Server: {r.json().get('model', 'unknown')}")
        print(f"  Speculation: {r.json().get('speculation_method', 'none')}")
        print()
    except Exception as e:
        print(f"  ERROR: Cannot reach server at {base_url}: {e}")
        return

    cursor_pos = len(REALISTIC_FILE) // 2

    def make_request(file_content=REALISTIC_FILE, cursor=cursor_pos, max_tokens=64):
        return {
            "file_path": "test/config.py",
            "file_content": file_content,
            "cursor_position": cursor,
            "max_tokens": max_tokens,
        }

    # --- Scenario 1: Cold start (no cache reuse) ---
    # Alternate between two very different files to force cache misses
    file_a = REALISTIC_FILE
    file_b = "# completely different file\n" * 200
    cursor_b = len(file_b) // 2

    def cold_request():
        # Send file_b first to invalidate cache, then file_a
        client.post("/v1/completions", json=make_request(file_b, cursor_b, max_tokens=4))
        t0 = time.perf_counter()
        client.post("/v1/completions", json=make_request(file_a, cursor_pos, max_tokens=16))
        return time.perf_counter() - t0

    times = [cold_request() for _ in range(iterations)]
    report("Cold start (cache miss)", times)

    # --- Scenario 2: Warm cache (same file, same cursor) ---
    # First request to warm the cache
    client.post("/v1/completions", json=make_request(max_tokens=4))

    def warm_request():
        t0 = time.perf_counter()
        client.post("/v1/completions", json=make_request(max_tokens=16))
        return time.perf_counter() - t0

    times = [warm_request() for _ in range(iterations)]
    report("Warm cache (same file+cursor)", times)

    # --- Scenario 3: Same file, cursor moves slightly ---
    def cursor_move_request():
        import random
        offset = random.randint(-50, 50)
        c = max(0, min(len(REALISTIC_FILE) - 1, cursor_pos + offset))
        t0 = time.perf_counter()
        client.post("/v1/completions", json=make_request(cursor=c, max_tokens=16))
        return time.perf_counter() - t0

    times = [cursor_move_request() for _ in range(iterations)]
    report("Cursor move (±50 chars)", times)

    # --- Scenario 4: Simple prompt mode ---
    def prompt_request():
        t0 = time.perf_counter()
        client.post("/v1/completions", json={
            "prompt": "def fibonacci(n):\n    ",
            "max_tokens": 32,
        })
        return time.perf_counter() - t0

    times = [prompt_request() for _ in range(iterations)]
    report("Simple prompt mode", times)

    # --- Print server stats ---
    print()
    try:
        r = client.get("/v1/stats")
        stats = r.json()
        print("  Server stats:")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for k2, v2 in v.items():
                    print(f"      {k2}: {v2}")
            else:
                print(f"    {k}: {v}")
    except Exception:
        pass

    client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Newsweep latency benchmarks")
    parser.add_argument("--mode", choices=["components", "http", "all"], default="components")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8741)
    args = parser.parse_args()

    if args.mode in ("components", "all"):
        bench_components(args.iterations)
    if args.mode in ("http", "all"):
        bench_http(args.iterations, args.host, args.port)


if __name__ == "__main__":
    main()
