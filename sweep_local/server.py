"""
FastAPI server that serves sweep-next-edit-v2-7B (MLX 4-bit) with a
Tabby-compatible completions API optimized for Apple Silicon.

Features:
- N-gram speculative decoding (no draft model needed, ~5x faster than baseline)
- Fallback to Qwen2.5-0.5B draft model speculation if n-gram disabled
- Token healing for partial-word accuracy (trie-based, sub-1ms)
- KV cache reuse across consecutive requests
- Early cancellation when edit is complete (suffix matching)
- Cross-file context from recent diffs for multi-file edit patterns
- PSI (type-resolved definitions) context from JetBrains
- Request cancellation for fast typing
- Latency/throughput metrics at /v1/stats

Start with: uvicorn sweep_local.server:app --host 0.0.0.0 --port 8741
"""

import collections
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
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field

from sweep_local.config import (
    DRAFT_MODEL_PATH,
    EARLY_STOP_MATCH_TOKENS,
    ENABLE_TOKEN_HEALING,
    HOST,
    MAX_CROSS_FILE_DIFFS,
    MAX_HEALING_TOKENS,
    MAX_NEW_TOKENS,
    MAX_PSI_DEFINITIONS,
    MAX_PSI_TOKENS,
    MODEL_ID,
    MODEL_PATH,
    NGRAM_N,
    NUM_DRAFT_TOKENS,
    NUM_LINES_AFTER,
    NUM_LINES_BEFORE,
    PORT,
    PROJECT_ROOT,
    USE_NGRAM_SPECULATION,
)
from sweep_local.file_watcher import diff_store, start_watcher
from sweep_local.sweep_prompt import (
    STOP_TOKENS,
    FileChunk,
    build_prompt,
    format_diff,
    is_pure_insertion_above_cursor,
)
from sweep_local.token_healing import (
    VocabTrie,
    find_healing_boundary,
    make_healing_processor,
    warmup_prefix_cache,
)

logger = logging.getLogger("newsweep")
logging.basicConfig(level=logging.INFO)

# --- Global state (initialized in load_model) ---
model = None
draft_model = None
tokenizer = None
sampler = None
stop_ids: set[int] = set()
vocab_trie: VocabTrie | None = None
observer = None

# KV cache for prompt prefix reuse between consecutive requests.
# Protected by _inference_lock — only one generation runs at a time.
# _last_cache_snapshot stores trimmed (keys, values, offset) tuples per layer
# so we avoid deepcopy of the full pre-allocated buffers.
_last_tokens: list[int] | None = None
_last_cache = None
_last_ngram_index: dict[tuple[int, ...], list[int]] | None = None

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
        self._healing_applied = 0
        self._healing_tokens_removed = 0
        self._psi_requests = 0
        self._psi_definitions_total = 0
        # Phase timing accumulators (rolling window of dicts)
        self._phase_timings: collections.deque[dict[str, float]] = collections.deque(
            maxlen=window
        )

    def record(self, latency: float, n_tokens: int, cache_hit: bool,
               draft_accepted: int = 0, draft_total: int = 0,
               healing_tokens: int = 0,
               phase_timings: dict[str, float] | None = None):
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
            if healing_tokens > 0:
                self._healing_applied += 1
                self._healing_tokens_removed += healing_tokens
            if phase_timings:
                self._phase_timings.append(phase_timings)

    def record_cancel(self):
        with self._lock:
            self._cancellations += 1

    def record_psi(self, num_definitions: int):
        with self._lock:
            self._psi_requests += 1
            self._psi_definitions_total += num_definitions

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

            # Compute p50 phase timings
            phase_p50: dict[str, float] = {}
            if self._phase_timings:
                all_phases: set[str] = set()
                for pt in self._phase_timings:
                    all_phases.update(pt.keys())
                for phase in sorted(all_phases):
                    vals = sorted(
                        pt.get(phase, 0.0) for pt in self._phase_timings
                    )
                    phase_p50[phase] = round(percentile(vals, 50) * 1000, 1)

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
                "healing_applied": self._healing_applied,
                "healing_tokens_removed": self._healing_tokens_removed,
                "psi_requests": self._psi_requests,
                "psi_avg_definitions": (
                    self._psi_definitions_total / self._psi_requests
                    if self._psi_requests > 0
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
                "phase_timings_p50_ms": phase_p50,
                "window_size": n,
            }


metrics = Metrics()


def load_model():
    global model, draft_model, tokenizer, sampler, stop_ids, vocab_trie
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
    # Build vocabulary trie for token healing
    if ENABLE_TOKEN_HEALING:
        vocab_trie = VocabTrie.from_tokenizer(tokenizer)
        warmup_prefix_cache(vocab_trie, tokenizer)
        logger.info("Token healing initialized")
    # Load draft model for speculative decoding (skipped when using n-gram speculation)
    if DRAFT_MODEL_PATH and not USE_NGRAM_SPECULATION:
        logger.info(f"Loading draft model {DRAFT_MODEL_PATH}...")
        draft_model, draft_tok = load(DRAFT_MODEL_PATH)
        if draft_tok.vocab_size != tokenizer.vocab_size:
            logger.warning(
                f"Draft vocab ({draft_tok.vocab_size}) != main vocab ({tokenizer.vocab_size}), "
                "disabling speculative decoding"
            )
            draft_model = None
    elif USE_NGRAM_SPECULATION:
        logger.info("Using n-gram speculative decoding (no draft model needed)")
    logger.info(f"Models loaded in {time.time() - t0:.1f}s")


def _warmup():
    """Run a short dummy generation to compile Metal kernels."""
    logger.info("Warming up...")
    t0 = time.time()
    dummy = mx.array(tokenizer.encode("def f():\n    "))
    if draft_model is not None and not USE_NGRAM_SPECULATION:
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


class PsiDefinition(BaseModel):
    """A type-resolved definition from the JetBrains Program Structure Interface."""
    kind: str                          # "class", "method", "field", "function", "type_alias"
    name: str                          # "DatabaseClient"
    qualified_name: str                # "com.example.db.DatabaseClient"
    signature: str                     # full signature text
    file_path: str | None = None       # source file, if resolvable
    body_preview: str | None = None    # first N lines of body (for short methods)
    priority: int = 0                  # higher = more important (0 = default)


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
    psi_context: list[PsiDefinition] | None = None


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


def _extend_ngram_index(
    index: dict[tuple[int, ...], list[int]],
    tokens: list[int],
    start: int,
    n: int = NGRAM_N,
) -> None:
    """Extend an existing n-gram index with new tokens starting at position start.

    Only adds entries whose predicted position (i + ctx_size) is > start,
    avoiding duplicates with entries already present from a previous build.
    """
    for ctx_size in range(1, n):
        for i in range(max(0, start - ctx_size + 1), len(tokens) - ctx_size):
            pos = i + ctx_size
            if pos <= start:
                continue
            key = tuple(tokens[i : i + ctx_size])
            if key not in index:
                index[key] = []
            index[key].append(pos)


def build_ngram_index(tokens: list[int], n: int = NGRAM_N) -> dict[tuple[int, ...], list[int]]:
    """
    Build an n-gram index mapping context tuples to their positions in the token list.
    Builds indices for all context sizes from 1 up to (n-1) to support fallback matching.
    """
    index: dict[tuple[int, ...], list[int]] = {}
    _extend_ngram_index(index, tokens, start=0, n=n)
    return index


def ngram_draft_tokens(
    ngram_index: dict[tuple[int, ...], list[int]],
    prompt_tokens: list[int],
    recent_tokens: list[int],
    num_draft: int,
    n: int = NGRAM_N,
) -> list[int]:
    """
    Propose draft tokens by finding the longest n-gram match in the prompt.
    Searches from the largest context size down to bigrams for the best match.
    Returns up to num_draft proposed tokens.
    """
    context_len = n - 1
    if len(recent_tokens) < context_len:
        return []

    for ctx_size in range(context_len, 0, -1):
        key = tuple(recent_tokens[-ctx_size:])
        positions = ngram_index.get(key, [])
        if positions:
            pos = positions[-1]
            draft = prompt_tokens[pos : pos + num_draft]
            return draft

    return []


def _snapshot_cache(cache: list) -> list[tuple]:
    """Save a lightweight snapshot of the KV cache (only the used portion).

    Returns a list of (keys_slice, values_slice, offset) tuples.
    The slices are contiguous copies of just the used region, avoiding
    the full pre-allocated buffer that KVCache maintains internally.
    """
    snapshot = []
    for layer in cache:
        used = layer.offset
        if layer.keys is not None and used > 0:
            # Slice + concatenate forces a contiguous copy of just the used region
            k = layer.keys[..., :used, :]
            v = layer.values[..., :used, :]
            snapshot.append((k, v, used))
        else:
            snapshot.append(None)
    # Force evaluation so the snapshot doesn't hold references to the full buffer
    arrays = []
    for s in snapshot:
        if s is not None:
            arrays.extend([s[0], s[1]])
    if arrays:
        mx.eval(*arrays)
    return snapshot


def _restore_cache(snapshot: list[tuple], trim_to: int | None = None) -> list:
    """Restore a KV cache from a snapshot.

    If trim_to is provided, only restore up to that many tokens.
    """
    cache = []
    for s in snapshot:
        c = KVCache()
        if s is not None:
            k, v, offset = s
            if trim_to is not None and trim_to < offset:
                k = k[..., :trim_to, :]
                v = v[..., :trim_to, :]
                offset = trim_to
            c.update_and_fetch(k, v)
        cache.append(c)
    return cache


def generate(
    prompt: str,
    max_tokens: int = MAX_NEW_TOKENS,
    seq: int = 0,
    suffix_tokens: list[int] | None = None,
) -> str:
    """
    Run inference with KV-cache reuse, request cancellation, optional n-gram
    speculative decoding, and early cancellation when the edit is complete.

    Thread-safe: only one generation runs at a time via _inference_lock.
    The lock also protects _last_tokens / _last_cache from concurrent access.

    Args:
        suffix_tokens: If provided, used for early cancellation — when the model
            produces EARLY_STOP_MATCH_TOKENS consecutive tokens matching this suffix,
            generation stops early and the remaining suffix is appended.
    """
    global _last_tokens, _last_cache, _last_ngram_index

    timings: dict[str, float] = {}

    t = time.perf_counter()
    tokens = tokenizer.encode(prompt)
    timings["tokenize"] = time.perf_counter() - t

    use_ngram = USE_NGRAM_SPECULATION and len(tokens) > NGRAM_N

    # Token healing: detect partial token boundary and constrain generation
    healing_prefix = ""
    healing_processor = None
    t = time.perf_counter()
    if ENABLE_TOKEN_HEALING and vocab_trie is not None and len(tokens) > 1:
        tokens, healing_prefix = find_healing_boundary(
            tokens, tokenizer, max_rollback=MAX_HEALING_TOKENS
        )
        if healing_prefix:
            healing_processor = make_healing_processor(
                healing_prefix, vocab_trie, tokenizer,
                prompt_token_count=len(tokens),
            )
            logger.debug(
                "Token healing: prefix=%r, rolled back to %d tokens",
                healing_prefix, len(tokens),
            )
    timings["healing"] = time.perf_counter() - t

    t_lock = time.perf_counter()
    with _inference_lock:
        timings["lock_wait"] = time.perf_counter() - t_lock

        # Check if we were already superseded while waiting for the lock
        if _request_seq > seq:
            metrics.record_cancel()
            return ""

        # --- Prepare cache ---
        reused = 0
        cache = None
        cache_hit = False

        t = time.perf_counter()
        if _last_tokens is not None and _last_cache is not None:
            common = _common_prefix_len(_last_tokens, tokens)
            common = min(common, len(tokens) - 1)
            if common > 0:
                try:
                    cache = _restore_cache(_last_cache, trim_to=common)
                    reused = common
                    cache_hit = True
                except Exception:
                    logger.warning("KV cache restore failed, starting fresh")
                    cache = None

        if cache is None:
            if draft_model is not None and not use_ngram:
                cache = make_prompt_cache(model) + make_prompt_cache(draft_model)
            else:
                cache = make_prompt_cache(model)
        timings["cache_reuse"] = time.perf_counter() - t

        prompt_array = mx.array(tokens[reused:])

        # --- Generate ---
        t0 = time.time()
        t_gen = time.perf_counter()
        generated: list[int] = []
        cancelled = False
        hit_stop = False
        early_stopped = False
        draft_accepted = 0
        draft_total = 0
        fwd_passes = 0
        ngram_index = None

        # Early cancellation state
        suffix_match_count = 0
        suffix_pos = 0

        try:
            if use_ngram:
                # --- N-gram speculative decoding (manual loop) ---
                # We cannot use generate_step here because n-gram verification
                # calls model() directly, which mutates the shared KV cache.
                # generate_step maintains its own internal state and would feed
                # stale tokens at wrong cache positions, producing garbage.
                t = time.perf_counter()
                # Reuse cached n-gram index when possible
                if (
                    _last_ngram_index is not None
                    and _last_tokens is not None
                    and cache_hit
                    and reused > 0
                ):
                    ngram_index = {
                        key: valid
                        for key, positions in _last_ngram_index.items()
                        if (valid := [p for p in positions if p < reused])
                    }
                    _extend_ngram_index(ngram_index, tokens, reused, NGRAM_N)
                else:
                    ngram_index = build_ngram_index(tokens)
                timings["ngram_index"] = time.perf_counter() - t
                all_tokens = list(tokens)

                # Process prompt (model expects 2D input: batch × seq_len)
                t = time.perf_counter()
                logits = model(prompt_array[None], cache=cache)
                mx.eval(logits)
                timings["prompt_eval"] = time.perf_counter() - t

                tokens_remaining = max_tokens
                # Healing completes within MAX_HEALING_TOKENS steps; after
                # that we stop building mx.array(all_tokens) each iteration.
                healing_steps_left = MAX_HEALING_TOKENS if healing_processor else 0
                while tokens_remaining > 0:
                    # Apply healing processor
                    step_logits = logits[0, -1:, :]
                    if healing_steps_left > 0:
                        step_logits = healing_processor(
                            mx.array(all_tokens), step_logits.squeeze(0)
                        )
                        step_logits = step_logits[None, :]
                        healing_steps_left -= 1

                    # Sample base token
                    tid = sampler(step_logits).item()

                    if _request_seq > seq:
                        cancelled = True
                        break
                    if tid in stop_ids:
                        hit_stop = True
                        break

                    generated.append(tid)
                    all_tokens.append(tid)
                    tokens_remaining -= 1

                    # Early cancellation check
                    if suffix_tokens and _check_early_stop(
                        tid, suffix_tokens, suffix_pos
                    ):
                        suffix_match_count += 1
                        suffix_pos += 1
                        if suffix_match_count >= EARLY_STOP_MATCH_TOKENS:
                            remaining = suffix_tokens[suffix_pos:]
                            generated.extend(remaining)
                            early_stopped = True
                            break
                    else:
                        suffix_match_count = 0
                        suffix_pos = 0

                    # Try n-gram draft for next tokens
                    draft = ngram_draft_tokens(
                        ngram_index, tokens, all_tokens[-NGRAM_N:],
                        NUM_DRAFT_TOKENS,
                    )

                    if draft and tokens_remaining > 0:
                        # Feed [base_token] + draft through model to verify.
                        # base_token hasn't been fed to model yet (we only
                        # sampled it from logits), so include it.
                        # logits[0,0,:] verifies draft[0], logits[0,i,:] verifies draft[i].
                        verify_input = mx.array([[tid] + draft])
                        logits = model(verify_input, cache=cache)
                        mx.eval(logits)
                        fwd_passes += 1

                        draft_total += len(draft)
                        accepted = 0

                        for i, d in enumerate(draft):
                            v = sampler(logits[0, i : i + 1, :]).item()
                            if v != d:
                                # Trim unverified draft tokens from cache
                                trim = len(draft) - i
                                for layer in cache:
                                    layer.trim(trim)
                                # Use model's prediction as next token
                                if v in stop_ids:
                                    hit_stop = True
                                else:
                                    generated.append(v)
                                    all_tokens.append(v)
                                    tokens_remaining -= 1
                                break

                            accepted += 1
                            generated.append(d)
                            all_tokens.append(d)
                            tokens_remaining -= 1

                            if d in stop_ids:
                                hit_stop = True
                                trim = len(draft) - i - 1
                                if trim > 0:
                                    for layer in cache:
                                        layer.trim(trim)
                                break

                            # Early cancellation check for draft tokens
                            if suffix_tokens and _check_early_stop(
                                d, suffix_tokens, suffix_pos
                            ):
                                suffix_match_count += 1
                                suffix_pos += 1
                                if suffix_match_count >= EARLY_STOP_MATCH_TOKENS:
                                    remaining = suffix_tokens[suffix_pos:]
                                    generated.extend(remaining)
                                    early_stopped = True
                                    trim = len(draft) - i - 1
                                    if trim > 0:
                                        for layer in cache:
                                            layer.trim(trim)
                                    break
                            else:
                                suffix_match_count = 0
                                suffix_pos = 0

                        draft_accepted += accepted

                        if hit_stop or early_stopped or _request_seq > seq:
                            if _request_seq > seq:
                                cancelled = True
                            break

                        # Set up logits for next iteration
                        if accepted == len(draft):
                            # All accepted: logits[0,-1,:] predicts next token
                            pass
                        else:
                            # Partial accept: need logits after the rejected/model token
                            last_tok = generated[-1]
                            logits = model(mx.array([[last_tok]]), cache=cache)
                            mx.eval(logits)
                            fwd_passes += 1
                    else:
                        # No draft — feed base token through model for next logits
                        logits = model(mx.array([[tid]]), cache=cache)
                        mx.eval(logits)
                        fwd_passes += 1

            elif draft_model is not None:
                # --- Draft-model speculative decoding (original path) ---
                lp = [healing_processor] if healing_processor else None
                for token_id, _, from_draft in speculative_generate_step(
                    prompt_array,
                    model,
                    draft_model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    num_draft_tokens=NUM_DRAFT_TOKENS,
                    logits_processors=lp,
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

                    # Early cancellation check
                    if suffix_tokens and _check_early_stop(
                        tid, suffix_tokens, suffix_pos
                    ):
                        suffix_match_count += 1
                        suffix_pos += 1
                        if suffix_match_count >= EARLY_STOP_MATCH_TOKENS:
                            remaining = suffix_tokens[suffix_pos:]
                            generated.extend(remaining)
                            early_stopped = True
                            break
                    else:
                        suffix_match_count = 0
                        suffix_pos = 0
            else:
                # --- Standard generation (no speculation) ---
                lp = [healing_processor] if healing_processor else None
                for token_id, _ in generate_step(
                    prompt_array,
                    model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    logits_processors=lp,
                ):
                    tid = token_id if isinstance(token_id, int) else token_id.item()
                    if _request_seq > seq:
                        cancelled = True
                        break
                    if tid in stop_ids:
                        hit_stop = True
                        break
                    generated.append(tid)

                    # Early cancellation check
                    if suffix_tokens and _check_early_stop(
                        tid, suffix_tokens, suffix_pos
                    ):
                        suffix_match_count += 1
                        suffix_pos += 1
                        if suffix_match_count >= EARLY_STOP_MATCH_TOKENS:
                            remaining = suffix_tokens[suffix_pos:]
                            generated.extend(remaining)
                            early_stopped = True
                            break
                    else:
                        suffix_match_count = 0
                        suffix_pos = 0

        except Exception:
            logger.exception("MLX generation failed")
            _last_tokens = None
            _last_cache = None
            _last_ngram_index = None
            return ""

        timings["generation"] = time.perf_counter() - t_gen
        elapsed = time.time() - t0

        # --- Save cache for next request ---
        # Don't save cache after early stop (suffix tokens weren't model-generated)
        # or after cancellation (incomplete generation).
        if not cancelled and not early_stopped:
            trim_gen = len(generated) + (1 if hit_stop else 0)
            if trim_gen > 0:
                for layer in cache:
                    layer.trim(min(trim_gen, layer.size()))
            _last_tokens = tokens
            _last_cache = _snapshot_cache(cache)
            _last_ngram_index = ngram_index if use_ngram else None

    # --- Record metrics ---
    n_gen = len(generated)
    healing_count = len(healing_prefix) if healing_prefix else 0

    t = time.perf_counter()
    completion = tokenizer.decode(generated, skip_special_tokens=False)

    # Strip healing prefix — the IDE already has this text
    if healing_prefix and completion.startswith(healing_prefix):
        completion = completion[len(healing_prefix):]

    for stop in STOP_TOKENS:
        idx = completion.find(stop)
        if idx >= 0:
            completion = completion[:idx]
    timings["decode"] = time.perf_counter() - t

    timings["total"] = elapsed
    metrics.record(
        elapsed, n_gen, cache_hit, draft_accepted, draft_total, healing_count,
        phase_timings=timings,
    )
    if cancelled:
        metrics.record_cancel()

    if n_gen > 0:
        tps = n_gen / elapsed if elapsed > 0 else 0
        model_tokens = n_gen
        if early_stopped and suffix_tokens:
            model_tokens = max(0, n_gen - max(0, len(suffix_tokens) - suffix_pos))
        gen_ms = timings.get("generation", 0) * 1000
        fwd_info = ""
        if use_ngram and fwd_passes > 0:
            ms_per_fwd = gen_ms / fwd_passes
            fwd_info = f" fwd={fwd_passes}x{ms_per_fwd:.0f}ms"
        logger.info(
            f"Generated {n_gen} tok ({model_tokens} model) in {elapsed:.3f}s "
            f"({tps:.0f} tok/s, reused={reused}, draft={draft_accepted}/{draft_total}"
            f"{fwd_info})"
            f"{' [early-stopped]' if early_stopped else ''}"
            f" | {' '.join(f'{k}={v*1000:.1f}ms' for k, v in timings.items())}"
        )

    return completion


def _check_early_stop(token_id: int, suffix_tokens: list[int], pos: int) -> bool:
    """Check if a generated token matches the expected suffix at the given position."""
    if pos >= len(suffix_tokens):
        return False
    return token_id == suffix_tokens[pos]


def psi_to_chunks(definitions: list[PsiDefinition]) -> list[FileChunk]:
    """Convert PSI definitions to FileChunks, sorted by priority and trimmed to budget.

    Higher priority definitions are kept first.  Definitions are dropped from the
    back (lowest priority) until the total estimated token count fits within
    MAX_PSI_TOKENS.
    """
    # Sort by priority descending, cap count
    defs = sorted(definitions, key=lambda d: d.priority, reverse=True)
    defs = defs[:MAX_PSI_DEFINITIONS]

    chunks: list[FileChunk] = []
    total_tokens = 0
    for d in defs:
        content = d.signature
        if d.body_preview:
            content += "\n" + d.body_preview
        # Rough token estimate: ~1 token per 3.5 chars
        est_tokens = len(content) // 3 + 10  # +10 for path/separator overhead
        if total_tokens + est_tokens > MAX_PSI_TOKENS:
            break
        path = d.file_path or f"definitions/{d.qualified_name}"
        chunks.append(FileChunk(file_path=path, content=content))
        total_tokens += est_tokens

    return chunks


def build_recent_changes(file_path: str) -> str:
    """Build the recent_changes prompt section from the diff store.

    Includes same-file diffs (up to 3) plus cross-file diffs from recently
    modified files (up to MAX_CROSS_FILE_DIFFS) to capture multi-file edit patterns.
    """
    parts = []

    # Same-file diffs
    diffs = diff_store.get_recent_diffs(file_path)
    for d in (diffs or [])[-3:]:
        if d.old_chunk != d.new_chunk:
            parts.append(format_diff(
                file_path=d.file_path,
                start_line=d.start_line,
                end_line=d.end_line,
                old_code=d.old_chunk,
                new_code=d.new_chunk,
            ))

    # Cross-file diffs (from other recently modified files)
    all_diffs = diff_store.get_recent_diffs(file_path=None)
    cross_file_count = 0
    for d in reversed(all_diffs):
        if cross_file_count >= MAX_CROSS_FILE_DIFFS:
            break
        if d.file_path == file_path:
            continue
        if d.old_chunk != d.new_chunk:
            parts.append(format_diff(
                file_path=d.file_path,
                start_line=d.start_line,
                end_line=d.end_line,
                old_code=d.old_chunk,
                new_code=d.new_chunk,
            ))
            cross_file_count += 1

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

    def _build_and_generate(file_path, file_content, cursor_position, changes_above_cursor):
        """Shared logic for building prompt and running generation."""
        t_ctx = time.perf_counter()
        recent_changes = build_recent_changes(file_path)

        # Convert PSI definitions to retrieval chunks
        retrieval_chunks = None
        if request.psi_context:
            retrieval_chunks = psi_to_chunks(request.psi_context)
            metrics.record_psi(len(request.psi_context))
        ctx_ms = (time.perf_counter() - t_ctx) * 1000

        t_bp = time.perf_counter()
        prompt, code_block, block_start, relative_cursor = build_prompt(
            file_path=file_path,
            file_contents=file_content,
            cursor_position=cursor_position,
            recent_changes=recent_changes,
            retrieval_chunks=retrieval_chunks,
            changes_above_cursor=changes_above_cursor,
            num_lines_before=NUM_LINES_BEFORE,
            num_lines_after=NUM_LINES_AFTER,
        )
        diff_store.seed_file(file_path, file_content)
        bp_ms = (time.perf_counter() - t_bp) * 1000

        # Compute suffix tokens for early cancellation:
        # the code block text after the cursor, tokenized
        t_st = time.perf_counter()
        suffix_text = code_block[relative_cursor:]
        stokens = tokenizer.encode(suffix_text) if suffix_text.strip() else None
        st_ms = (time.perf_counter() - t_st) * 1000

        logger.info(
            f"Pre-generate: context={ctx_ms:.1f}ms prompt={bp_ms:.1f}ms suffix_tok={st_ms:.1f}ms"
        )

        completion = generate(prompt, request.max_tokens, seq, suffix_tokens=stokens)
        if completion and is_pure_insertion_above_cursor(
            code_block, completion, relative_cursor
        ):
            completion = ""
        return completion

    if request.segments is not None:
        prefix = request.segments.prefix
        suffix = request.segments.suffix or ""
        filepath = request.segments.filepath or "unknown"

        if request.file_content and request.cursor_position is not None:
            completion = _build_and_generate(
                filepath, request.file_content,
                request.cursor_position, request.changes_above_cursor,
            )
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
        completion = _build_and_generate(
            request.file_path, request.file_content,
            request.cursor_position, request.changes_above_cursor,
        )
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
        "speculative_decoding": USE_NGRAM_SPECULATION or draft_model is not None,
        "speculation_method": (
            "ngram" if USE_NGRAM_SPECULATION
            else ("draft_model" if draft_model else "none")
        ),
        "token_healing": ENABLE_TOKEN_HEALING,
        "early_cancellation": True,
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
