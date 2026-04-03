# newsweep Documentation

Welcome to the newsweep documentation. newsweep is a local next-edit autocomplete server that runs on Apple Silicon, predicting your *next edit* rather than just completing the current line.

## How to read these docs

### [Getting Started](getting-started/)

Setup, configuration, and IDE integration.

| Document | Description |
|---|---|
| [Installation](getting-started/installation.md) | Installing newsweep and downloading models |
| [Configuration](getting-started/configuration.md) | Tuning inference, speculation, and context |
| [IDE Setup](getting-started/ide-setup.md) | Connecting JetBrains IDEs via Tabby |

### [Architecture](architecture/)

How the system works, end to end.

| Document | Description |
|---|---|
| [Overview](architecture/overview.md) | Request flow, modules, and concurrency model |
| [Prompt Construction](architecture/prompt-construction.md) | How the model prompt is built from code and diffs |
| [Inference Pipeline](architecture/inference-pipeline.md) | The generation loop: from tokens in to completion out |

### [Deep Dives](deep-dives/)

Detailed explanations of each optimization technique.

| Document | Description |
|---|---|
| [N-gram Speculation](deep-dives/ngram-speculation.md) | Speculative decoding without a draft model |
| [Token Healing](deep-dives/token-healing.md) | Fixing partial-word tokenization boundaries |
| [KV Cache Reuse](deep-dives/kv-cache-reuse.md) | Sharing computation across consecutive requests |
| [Early Cancellation](deep-dives/early-cancellation.md) | Stopping generation when the edit is complete |
| [File Watcher](deep-dives/file-watcher.md) | Tracking file changes for diff context |

### [API](api/)

Endpoint reference for integration.

| Document | Description |
|---|---|
| [Endpoints](api/endpoints.md) | Complete API reference with request/response schemas |

### [Benchmarks](benchmarks/)

Performance and quality measurement.

| Document | Description |
|---|---|
| [Latency Benchmark](benchmarks/latency.md) | Component and HTTP latency profiling |
| [Quality Benchmark](benchmarks/quality.md) | Prediction accuracy against realistic editing scenarios |
