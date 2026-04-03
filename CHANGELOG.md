# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-02

### Added

- N-gram speculative decoding: replaces draft model with prompt-based n-gram lookup (~5x faster, no extra memory)
- Token healing: multi-token healing with vocabulary trie for sub-1ms partial-word correction
- Early cancellation: stops generation when output re-aligns with known suffix
- Cross-file context: includes recent diffs from other project files in prompt
- PSI (Program Structure Interface) context: accepts type-resolved definitions from JetBrains
- Healing and PSI metrics in `/v1/stats`
- Token healing and early cancellation status in `/v1/health`

### Changed

- Draft model (Qwen2.5-0.5B) is now optional -- only loaded when `USE_NGRAM_SPECULATION = False`
- Completions endpoint refactored to eliminate duplicated prompt-building logic
- README rewritten with architecture explanation and key techniques documentation

## [0.1.0] - 2026-04-02

### Added

- FastAPI server with Tabby-compatible `/v1/completions` endpoint
- sweep-next-edit-v2-7B (MLX 4-bit) for next-edit prediction
- Speculative decoding with Qwen2.5-0.5B draft model
- KV cache reuse across consecutive requests
- Request cancellation for fast typing
- File watcher that tracks recent diffs for prompt context
- Latency and throughput metrics at `/v1/stats`
- launchd plist for auto-start on macOS
- Tabby configuration example
