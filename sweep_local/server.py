"""
FastAPI server that serves sweep-next-edit-v2-7B (MLX 4-bit) with a
Tabby-compatible completions API optimized for Apple Silicon.

Features:
- Speculative decoding with Qwen2.5-0.5B draft model (~1.2x speedup)
- KV cache reuse across consecutive requests
- Request cancellation for fast typing
- Latency/throughput metrics at /v1/stats

Start with: uvicorn sweep_local.server:app --host 0.0.0.0 --port 8741
"""

import collections
import copy
import logging
import platform
import threading
import time
import uuid
from contextlib import asynccontextmanager

import mlx.core as mx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from mlx_lm import load
from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field

from sweep_local.config import (
    DRAFT_MODEL_PATH,
    HOST,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_PATH,
    NUM_DRAFT_TOKENS,
    NUM_LINES_AFTER,
    NUM_LINES_BEFORE,
    PORT,
    PROJECT_ROOT,
)
from sweep_local.file_watcher import diff_store, start_watcher
from sweep_local.sweep_prompt import (
    STOP_TOKENS,
    build_prompt,
    format_diff,
    is_pure_insertion_above_cursor,
)

logger = logging.getLogger("newsweep")
logging.basicConfig(level=logging.INFO)

# --- Global state (initialized in load_model) ---
model = None
draft_model = None
tokenizer = None
sampler = None
stop_ids: set[int] = set()
observer = None

# KV cache for prompt prefix reuse between consecutive requests.
# Protected by _inference_lock — only one generation runs at a time.
_last_tokens: list[int] | None = None
_last_cache = None

# Request cancellation: newer requests cause in-flight generation to stop early.
_request_seq = 0
_seq_lock = threading.Lock()

# Only one generation can run at a time (MLX is single-device).
# This also protects _last_tokens / _last_cache from races.
_inference_lock = threading.Lock()


# --- Metrics ---
class Metrics:
    """Thread-safe request metrics with a rolling window."""

    def __init__(self, window: int = 200):
        self._lock = threading.Lock()
        self._latencies: collections.deque[float] = collections.deque(maxlen=window)
        self._tokens_per_sec: collections.deque[float] = collections.deque(maxlen=window)
        self._total_requests = 0
        self._total_tokens = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cancellations = 0
        self._draft_accepted = 0
        self._draft_total = 0

    def record(self, latency: float, n_tokens: int, cache_hit: bool,
               draft_accepted: int = 0, draft_total: int = 0):
        with self._lock:
            self._latencies.append(latency)
            if n_tokens > 0 and latency > 0:
                self._tokens_per_sec.append(n_tokens / latency)
            self._total_requests += 1
            self._total_tokens += n_tokens
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            self._draft_accepted += draft_accepted
            self._draft_total += draft_total

    def record_cancel(self):
        with self._lock:
            self._cancellations += 1

    def snapshot(self) -> dict:
        with self._lock:
            lats = sorted(self._latencies)
            tps = sorted(self._tokens_per_sec)
            n = len(lats)

            def percentile(vals, p):
                if not vals:
                    return 0.0
                idx = int(len(vals) * p / 100)
                return vals[min(idx, len(vals) - 1)]

            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "cancellations": self._cancellations,
                "cache_hit_rate": (
                    self._cache_hits / (self._cache_hits + self._cache_misses)
                    if (self._cache_hits + self._cache_misses) > 0
                    else 0.0
                ),
                "draft_acceptance_rate": (
                    self._draft_accepted / self._draft_total
                    if self._draft_total > 0
                    else 0.0
                ),
                "latency_ms": {
                    "p50": round(percentile(lats, 50) * 1000, 1),
                    "p95": round(percentile(lats, 95) * 1000, 1),
                    "p99": round(percentile(lats, 99) * 1000, 1),
                },
                "tokens_per_sec": {
                    "p50": round(percentile(tps, 50), 1),
                    "p95": round(percentile(tps, 95), 1),
                },
                "window_size": n,
            }


metrics = Metrics()


def load_model():
    global model, draft_model, tokenizer, sampler, stop_ids
    logger.info(f"Loading {MODEL_PATH} (MLX 4-bit)...")
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    sampler = make_sampler(temp=0.0)
    # Pre-compute stop token IDs once
    vocab = tokenizer.get_vocab()
    stop_ids = {
        tokenizer.convert_tokens_to_ids(t)
        for t in STOP_TOKENS
        if t in vocab
    }
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    # Load draft model for speculative decoding
    if DRAFT_MODEL_PATH:
        logger.info(f"Loading draft model {DRAFT_MODEL_PATH}...")
        draft_model, draft_tok = load(DRAFT_MODEL_PATH)
        if draft_tok.vocab_size != tokenizer.vocab_size:
            logger.warning(
                f"Draft vocab ({draft_tok.vocab_size}) != main vocab ({tokenizer.vocab_size}), "
                "disabling speculative decoding"
            )
            draft_model = None
    logger.info(f"Models loaded in {time.time() - t0:.1f}s")


def _warmup():
    """Run a short dummy generation to compile Metal kernels."""
    logger.info("Warming up...")
    t0 = time.time()
    dummy = mx.array(tokenizer.encode("def f():\n    "))
    if draft_model is not None:
        for _ in speculative_generate_step(
            dummy, model, draft_model,
            max_tokens=4, sampler=sampler, num_draft_tokens=NUM_DRAFT_TOKENS,
        ):
            pass
    else:
        for _ in generate_step(dummy, model, max_tokens=4, sampler=sampler):
            pass
    logger.info(f"Warm-up done in {time.time() - t0:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global observer
    load_model()
    _warmup()
    observer = start_watcher(PROJECT_ROOT)
    logger.info(f"Watching {PROJECT_ROOT} for file changes")
    yield
    if observer:
        observer.stop()


app = FastAPI(title="newsweep", lifespan=lifespan)


# --- Tabby-compatible request/response models ---


class Segments(BaseModel):
    prefix: str
    suffix: str | None = None
    filepath: str | None = None
    git_url: str | None = None
    declarations: list | None = None
    relevant_snippets_from_changed_files: list | None = None
    relevant_snippets_from_recently_opened_files: list | None = None
    clipboard: str | None = None


class TabbyCompletionRequest(BaseModel):
    language: str | None = None
    segments: Segments | None = None
    temperature: float | None = None
    seed: int | None = None
    prompt: str | None = None
    max_tokens: int = Field(default=MAX_NEW_TOKENS, le=2048)
    stop: list[str] | None = None
    file_path: str | None = None
    file_content: str | None = None
    cursor_position: int | None = None
    changes_above_cursor: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0


class TabbyCompletionResponse(BaseModel):
    id: str
    choices: list[CompletionChoice]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common token prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def generate(prompt: str, max_tokens: int = MAX_NEW_TOKENS, seq: int = 0) -> str:
    """
    Run inference with speculative decoding, KV-cache reuse, and request cancellation.

    Thread-safe: only one generation runs at a time via _inference_lock.
    The lock also protects _last_tokens / _last_cache from concurrent access.
    """
    global _last_tokens, _last_cache

    tokens = tokenizer.encode(prompt)

    with _inference_lock:
        # Check if we were already superseded while waiting for the lock
        if _request_seq > seq:
            metrics.record_cancel()
            return ""

        # --- Prepare cache ---
        reused = 0
        cache = None
        cache_hit = False

        # Speculative decoding uses a combined cache [model_layers + draft_layers],
        # so we can only reuse it when both models' layer counts match the saved cache.
        if _last_tokens is not None and _last_cache is not None:
            common = _common_prefix_len(_last_tokens, tokens)
            common = min(common, len(tokens) - 1)
            if common > 0:
                try:
                    cache = copy.deepcopy(_last_cache)
                    trim_amount = cache[0].size() - common
                    if trim_amount > 0:
                        for layer in cache:
                            layer.trim(trim_amount)
                    reused = common
                    cache_hit = True
                except Exception:
                    logger.warning("KV cache deepcopy failed, starting fresh")
                    cache = None

        if cache is None:
            if draft_model is not None:
                cache = make_prompt_cache(model) + make_prompt_cache(draft_model)
            else:
                cache = make_prompt_cache(model)

        prompt_tokens = mx.array(tokens[reused:])

        # --- Generate ---
        t0 = time.time()
        generated = []
        cancelled = False
        hit_stop = False
        draft_accepted = 0
        draft_total = 0

        try:
            if draft_model is not None:
                for token_id, _, from_draft in speculative_generate_step(
                    prompt_tokens,
                    model,
                    draft_model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    num_draft_tokens=NUM_DRAFT_TOKENS,
                ):
                    tid = token_id if isinstance(token_id, int) else token_id.item()
                    draft_total += 1
                    if from_draft:
                        draft_accepted += 1
                    if _request_seq > seq:
                        cancelled = True
                        break
                    if tid in stop_ids:
                        hit_stop = True
                        break
                    generated.append(tid)
            else:
                for token_id, _ in generate_step(
                    prompt_tokens,
                    model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                ):
                    if _request_seq > seq:
                        cancelled = True
                        break
                    if token_id in stop_ids:
                        hit_stop = True
                        break
                    generated.append(token_id)
        except Exception:
            logger.exception("MLX generation failed")
            _last_tokens = None
            _last_cache = None
            return ""

        elapsed = time.time() - t0

        # --- Save cache for next request ---
        if not cancelled:
            trim_gen = len(generated) + (1 if hit_stop else 0)
            if trim_gen > 0:
                for layer in cache:
                    layer.trim(min(trim_gen, layer.size()))
            _last_tokens = tokens
            _last_cache = cache

    # --- Record metrics ---
    n_gen = len(generated)
    metrics.record(elapsed, n_gen, cache_hit, draft_accepted, draft_total)
    if cancelled:
        metrics.record_cancel()

    if n_gen > 0:
        tps = n_gen / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Generated {n_gen} tokens in {elapsed:.3f}s "
            f"({tps:.0f} tok/s, reused {reused}, draft {draft_accepted}/{draft_total})"
        )

    completion = tokenizer.decode(generated)

    for stop in STOP_TOKENS:
        idx = completion.find(stop)
        if idx >= 0:
            completion = completion[:idx]

    return completion


def build_recent_changes(file_path: str) -> str:
    """Build the recent_changes prompt section from the diff store."""
    diffs = diff_store.get_recent_diffs(file_path)
    if not diffs:
        return ""
    parts = []
    for d in diffs[-3:]:
        if d.old_chunk != d.new_chunk:
            parts.append(format_diff(
                file_path=d.file_path,
                start_line=d.start_line,
                end_line=d.end_line,
                old_code=d.old_chunk,
                new_code=d.new_chunk,
            ))
    return "\n".join(parts)


# Plain `def` → FastAPI runs in threadpool, event loop stays free for health/events.
@app.post("/v1/completions")
def completions(request: TabbyCompletionRequest):
    """Tabby-compatible completion endpoint."""
    global _request_seq

    with _seq_lock:
        _request_seq += 1
        seq = _request_seq

    completion_id = f"cmpl-{uuid.uuid4()}"

    if request.segments is not None:
        prefix = request.segments.prefix
        suffix = request.segments.suffix or ""
        filepath = request.segments.filepath or "unknown"

        if request.file_content and request.cursor_position is not None:
            recent_changes = build_recent_changes(filepath)
            prompt, code_block, block_start, relative_cursor = build_prompt(
                file_path=filepath,
                file_contents=request.file_content,
                cursor_position=request.cursor_position,
                recent_changes=recent_changes,
                changes_above_cursor=request.changes_above_cursor,
                num_lines_before=NUM_LINES_BEFORE,
                num_lines_after=NUM_LINES_AFTER,
            )
            diff_store.seed_file(filepath, request.file_content)
            completion = generate(prompt, request.max_tokens, seq)
            if completion and is_pure_insertion_above_cursor(
                code_block, completion, relative_cursor
            ):
                completion = ""
        else:
            completion = generate(prefix, request.max_tokens, seq)
            if suffix:
                for i in range(min(len(completion), len(suffix)), 0, -1):
                    if completion.endswith(suffix[:i]):
                        completion = completion[:-i]
                        break

        return TabbyCompletionResponse(
            id=completion_id,
            choices=[CompletionChoice(text=completion)],
        )

    elif request.file_path and request.file_content and request.cursor_position is not None:
        recent_changes = build_recent_changes(request.file_path)
        prompt, code_block, block_start, relative_cursor = build_prompt(
            file_path=request.file_path,
            file_contents=request.file_content,
            cursor_position=request.cursor_position,
            recent_changes=recent_changes,
            changes_above_cursor=request.changes_above_cursor,
            num_lines_before=NUM_LINES_BEFORE,
            num_lines_after=NUM_LINES_AFTER,
        )
        diff_store.seed_file(request.file_path, request.file_content)
        completion = generate(prompt, request.max_tokens, seq)
        if completion and is_pure_insertion_above_cursor(code_block, completion, relative_cursor):
            completion = ""

        return TabbyCompletionResponse(
            id=completion_id,
            choices=[CompletionChoice(text=completion)],
        )

    else:
        prompt = request.prompt or ""
        completion = generate(prompt, request.max_tokens, seq)
        return TabbyCompletionResponse(
            id=completion_id,
            choices=[CompletionChoice(text=completion)],
        )


@app.get("/v1/stats")
async def stats():
    """Latency and throughput metrics."""
    return metrics.snapshot()


@app.get("/v1/health")
async def health():
    return {
        "model": MODEL_ID,
        "device": "mlx",
        "speculative_decoding": draft_model is not None,
        "cuda_devices": [],
        "models": {
            "completion": {
                "Local": {
                    "model_id": MODEL_ID,
                    "device": "mlx",
                    "cuda_devices": [],
                }
            },
            "chat": None,
            "embedding": None,
        },
        "arch": platform.machine(),
        "cpu_info": platform.processor() or "unknown",
        "cpu_count": 1,
        "version": {
            "build_date": "2025-01-01",
            "build_timestamp": "2025-01-01T00:00:00Z",
            "git_sha": "0000000",
            "git_describe": "v0.21.0",
        },
    }


@app.post("/v1/health")
async def health_post():
    return await health()


@app.post("/v1/events")
async def events(request: Request):
    return PlainTextResponse("")


@app.get("/v1beta/server_setting")
async def server_setting():
    return {"disable_client_side_telemetry": True}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "sweepai",
            }
        ],
    }


def main():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
