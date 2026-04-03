# API Endpoints

newsweep exposes a Tabby-compatible API on port 8741 (configurable in `config.py`).

## POST /v1/completions

The main completion endpoint. Accepts three request formats.

### Format 1: Tabby segments (standard)

Used by the Tabby IDE plugin. Sends code before and after the cursor as separate fields.

```json
{
  "segments": {
    "prefix": "def compute(values):\n    total = sum(",
    "suffix": ")\n    return total / len(values)",
    "filepath": "/path/to/file.py"
  },
  "language": "python",
  "max_tokens": 128
}
```

### Format 2: Full file content (enhanced)

Sends the entire file and cursor position. Enables next-edit features (diff context, code block extraction, early cancellation).

```json
{
  "segments": {
    "prefix": "...",
    "suffix": "...",
    "filepath": "/path/to/file.py"
  },
  "file_content": "full file text here...",
  "cursor_position": 142,
  "changes_above_cursor": false,
  "psi_context": [
    {
      "kind": "method",
      "name": "compute",
      "qualified_name": "mypackage.math.compute",
      "signature": "def compute(values: list[float]) -> float",
      "file_path": "/path/to/math.py",
      "body_preview": "    total = sum(values)\n    return total / len(values)",
      "priority": 5
    }
  ]
}
```

### Format 3: Raw prompt

Direct prompt string. No code intelligence, no diff context.

```json
{
  "prompt": "def hello():\n    print(",
  "max_tokens": 64
}
```

### Request fields

| Field | Type | Default | Description |
|---|---|---|---|
| `segments` | object | `null` | Tabby segments (prefix, suffix, filepath) |
| `segments.prefix` | string | required | Code before cursor |
| `segments.suffix` | string | `""` | Code after cursor |
| `segments.filepath` | string | `"unknown"` | File path |
| `language` | string | `null` | Language identifier |
| `prompt` | string | `null` | Raw prompt (used if no segments) |
| `max_tokens` | int | `1024` | Max tokens to generate (capped at 2048) |
| `file_content` | string | `null` | Full file text (enables next-edit features) |
| `cursor_position` | int | `null` | Byte offset of cursor in file_content |
| `changes_above_cursor` | bool | `false` | Insertion mode (prefill only first line) |
| `psi_context` | array | `null` | Type-resolved definitions from IDE |

### PSI context fields

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | string | yes | `"class"`, `"method"`, `"field"`, `"function"`, `"type_alias"` |
| `name` | string | yes | Symbol name |
| `qualified_name` | string | yes | Fully qualified name |
| `signature` | string | yes | Full signature text |
| `file_path` | string | no | Source file path |
| `body_preview` | string | no | First N lines of body |
| `priority` | int | `0` | Higher = more important (sorted descending) |

### Response

```json
{
  "id": "cmpl-550e8400-e29b-41d4-a716-446655440000",
  "choices": [
    {
      "text": "values)\n    return total / len(values)",
      "index": 0
    }
  ]
}
```

An empty `text` means no completion was generated (request was cancelled, model produced a stop token immediately, or the completion was rejected as a pure insertion above cursor).

---

## GET /v1/health

Server health and model information.

```json
{
  "model": "sweepai/sweep-next-edit-v2-7B",
  "device": "mlx",
  "speculative_decoding": true,
  "speculation_method": "ngram",
  "cuda_devices": [],
  "models": {
    "completion": {
      "Local": {
        "model_id": "sweepai/sweep-next-edit-v2-7B",
        "device": "mlx",
        "cuda_devices": []
      }
    },
    "chat": null,
    "embedding": null
  },
  "arch": "arm64",
  "cpu_info": "Apple M2 Pro",
  "cpu_count": 1,
  "version": { ... }
}
```

Also available as `POST /v1/health` (for Tabby compatibility).

---

## GET /v1/stats

Rolling metrics over the last 200 requests.

```json
{
  "total_requests": 1523,
  "total_tokens": 45690,
  "cancellations": 312,
  "cache_hit_rate": 0.72,
  "draft_acceptance_rate": 0.68,
  "healing_applied": 89,
  "healing_tokens_removed": 134,
  "latency_ms": {
    "p50": 245.3,
    "p95": 512.7,
    "p99": 891.2
  },
  "tokens_per_sec": {
    "p50": 38.2,
    "p95": 52.1
  },
  "window_size": 200
}
```

| Metric | Description |
|---|---|
| `total_requests` | Total completion requests since server start |
| `total_tokens` | Total tokens generated |
| `cancellations` | Requests cancelled by newer requests |
| `cache_hit_rate` | Fraction of requests that reused the KV cache |
| `draft_acceptance_rate` | Fraction of speculative draft tokens accepted |
| `healing_applied` | Requests where token healing was active |
| `healing_tokens_removed` | Total tokens rolled back by healing |
| `latency_ms` | Generation latency percentiles |
| `tokens_per_sec` | Throughput percentiles |
| `window_size` | Number of requests in the rolling window |

---

## GET /v1/models

List loaded models (Tabby/OpenAI-compatible format).

```json
{
  "object": "list",
  "data": [
    {
      "id": "sweepai/sweep-next-edit-v2-7B",
      "object": "model",
      "owned_by": "sweepai"
    }
  ]
}
```

---

## POST /v1/events

No-op endpoint for Tabby telemetry events. Always returns an empty response.

---

## GET /v1beta/server_setting

Returns server settings. Currently only disables client-side telemetry:

```json
{
  "disable_client_side_telemetry": true
}
```
