# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
