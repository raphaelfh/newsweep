# Installation

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- ~6 GB disk space for models
- ~8 GB unified memory during inference

## 1. Clone and install

```bash
git clone https://github.com/raphaelfh/newsweep.git
cd newsweep
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, ruff, and pre-commit for development.

## 2. Download and convert models

newsweep uses two models, both quantized to 4-bit MLX format. The `mlx_lm` CLI (installed with the project) handles downloading from Hugging Face and quantizing in one command.

### Main model: sweep-next-edit-v2-7B

This is the core model -- a 7B parameter LLM fine-tuned by [SweepAI](https://github.com/sweepai) specifically for next-edit prediction. It understands a custom prompt format with file separators, cursor markers, and diff context.

```bash
mlx_lm.convert \
  --hf-path sweepai/sweep-next-edit-v2-7B \
  -q --q-bits 4 \
  --mlx-path models/sweep-next-edit-v2-7B-4bit
```

This downloads ~14 GB of weights, quantizes to ~4 GB, and saves MLX-format tensors to `models/sweep-next-edit-v2-7B-4bit/`.

### Draft model: Qwen2.5-0.5B (optional)

A small general-purpose LLM used for [draft-model speculative decoding](../deep-dives/ngram-speculation.md). Only needed if you disable n-gram speculation (which is the default and usually faster).

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-0.5B \
  -q --q-bits 4 \
  --mlx-path models/qwen2.5-0.5b-4bit
```

### What `mlx_lm.convert` does

It downloads PyTorch/safetensors weights from Hugging Face, converts them to Apple's [MLX](https://github.com/ml-explore/mlx) tensor format (optimized for unified memory on Apple Silicon), and applies 4-bit quantization. Each output folder is self-contained: weights, tokenizer, and config.

## 3. Start the server

```bash
./start.sh
# or
newsweep
```

The server starts on `http://localhost:8741`. Verify with:

```bash
curl http://localhost:8741/v1/health
```

## 4. Auto-start with launchd (optional)

To have newsweep start automatically at login:

1. Edit `com.sweep.local.plist` -- update the Python path, working directory, and log paths to match your environment.

2. Install and load:

```bash
cp com.sweep.local.plist ~/Library/LaunchAgents/com.newsweep.local.plist
launchctl load ~/Library/LaunchAgents/com.newsweep.local.plist
```

3. Manage the service:

```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.newsweep.local.plist

# Restart (unload + load)
launchctl unload ~/Library/LaunchAgents/com.newsweep.local.plist
launchctl load ~/Library/LaunchAgents/com.newsweep.local.plist
```

Logs are written to `logs/stdout.log` and `logs/stderr.log`.

## Running tests

```bash
pytest                            # all tests
pytest tests/test_sweep_prompt.py # single file
pytest -k test_name               # single test by name
```

## Linting

```bash
ruff check .             # lint
ruff format --check .    # format check
ruff format .            # auto-format
```

Pre-commit hooks run ruff automatically on each commit if you install them:

```bash
pre-commit install
```
