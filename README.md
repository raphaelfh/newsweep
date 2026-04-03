# newsweep

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black.svg)](https://support.apple.com/en-us/116943)

Local next-edit autocomplete server for JetBrains IDEs via [Tabby](https://tabby.tabbyml.com/), powered by [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B) running on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## What it does

newsweep runs a local inference server that provides intelligent code completions by predicting your *next edit* rather than just completing the current line. It watches your project files for changes and uses recent diffs as context, so suggestions are aware of what you've been working on.

Key features:

- **Next-edit prediction** -- suggests edits based on your recent changes, not just cursor position
- **Speculative decoding** -- uses a small Qwen2.5-0.5B draft model for ~1.2x faster generation
- **KV cache reuse** -- reuses computation across consecutive requests for lower latency
- **Request cancellation** -- fast typing cancels in-flight requests so the server stays responsive
- **Tabby-compatible API** -- works with the Tabby plugin for PyCharm (and other JetBrains IDEs)

## How it works

```
 JetBrains IDE (Tabby plugin)
        │
        │  POST /v1/completions
        ▼
 ┌─────────────────────────────────────────────┐
 │  newsweep server (FastAPI)                   │
 │                                              │
 │  1. File Watcher ──► tracks recent diffs     │
 │  2. Prompt Builder ──► builds context from:  │
 │     • code around cursor                     │
 │     • recent changes (diffs)                 │
 │     • file context                           │
 │  3. MLX Inference ──► generates completion   │
 │     • sweep-next-edit-v2-7B (main model)     │
 │     • qwen2.5-0.5B (draft, speculative)      │
 │     • KV cache reuse between requests        │
 └─────────────────────────────────────────────┘
```

The server watches your project directory for file saves. When you edit code, recent diffs are recorded and injected into the model prompt as "recent changes" context. This allows the model to predict what you're likely to edit *next*, not just complete the current token.

Speculative decoding runs a small draft model (Qwen2.5-0.5B) ahead of the main model, proposing candidate tokens that the main model verifies in batch. When the draft model guesses correctly (which is often for boilerplate), you get multiple tokens per forward pass.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- ~6 GB of disk space for models
- ~8 GB of unified memory during inference

## Setup

1. Clone the repository:

```bash
git clone https://github.com/raphaelfh/newsweep.git
cd newsweep
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

3. Download and convert the models.

   newsweep requires two models, both converted to MLX 4-bit format. The `mlx_lm` CLI (installed with the project dependencies in step 2) handles downloading from Hugging Face and quantizing in one command.

   **Main model** -- [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B), the next-edit prediction model by SweepAI. This is the only project-specific model; everything else is standard tooling.

   ```bash
   mlx_lm.convert \
     --hf-path sweepai/sweep-next-edit-v2-7B \
     -q --q-bits 4 \
     --mlx-path models/sweep-next-edit-v2-7B-4bit
   ```

   This downloads the original model weights from Hugging Face (~14 GB), quantizes them to 4-bit precision (~4 GB), and saves the result in MLX format to `models/sweep-next-edit-v2-7B-4bit/`.

   **Draft model** -- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), a small general-purpose language model by the Qwen team. It's used as the "draft" model for [speculative decoding](https://arxiv.org/abs/2302.01318) -- it proposes candidate tokens that the main model verifies in batch, which speeds up generation by ~1.2x. This model is not specific to newsweep; any small LM would work, but Qwen2.5-0.5B is a good balance of speed and quality.

   ```bash
   mlx_lm.convert \
     --hf-path Qwen/Qwen2.5-0.5B \
     -q --q-bits 4 \
     --mlx-path models/qwen2.5-0.5b-4bit
   ```

   After both commands finish, your `models/` directory should contain:

   ```
   models/
     sweep-next-edit-v2-7B-4bit/
       config.json
       model.safetensors
       model.safetensors.index.json
       tokenizer.json
       tokenizer_config.json
     qwen2.5-0.5b-4bit/
       config.json
       model.safetensors
       model.safetensors.index.json
       tokenizer.json
       tokenizer_config.json
   ```

   > **What `mlx_lm.convert` does:** It downloads the original PyTorch/safetensors weights from Hugging Face, converts them to Apple's [MLX](https://github.com/ml-explore/mlx) tensor format (optimized for Apple Silicon unified memory), and applies 4-bit quantization to reduce memory usage. The converted models are self-contained -- each folder has the weights, tokenizer, and config needed to run inference.

4. Start the server:

```bash
./start.sh
# or
newsweep
```

The server runs on `http://localhost:8741`.

## macOS setup (auto-start with launchd)

To have newsweep start automatically at login, install the included launchd plist:

1. Edit `com.sweep.local.plist` and update the paths to match your environment (Python path, working directory, log paths).

2. Copy it to `~/Library/LaunchAgents/`:

```bash
cp com.sweep.local.plist ~/Library/LaunchAgents/com.newsweep.local.plist
```

3. Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.newsweep.local.plist
```

4. Verify it's running:

```bash
curl http://localhost:8741/v1/health
```

To stop or restart:

```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.newsweep.local.plist

# Restart
launchctl unload ~/Library/LaunchAgents/com.newsweep.local.plist
launchctl load ~/Library/LaunchAgents/com.newsweep.local.plist
```

Logs are written to `logs/stdout.log` and `logs/stderr.log`.

## JetBrains IDE setup (PyCharm, IntelliJ, WebStorm, etc.)

### Step 1: Install the Tabby plugin

1. Open your JetBrains IDE.
2. Go to **Settings** > **Plugins** > **Marketplace**.
3. Search for **Tabby** and click **Install**.
4. Restart the IDE.

### Step 2: Configure Tabby to use newsweep

1. Copy the included Tabby config to your home directory:

```bash
mkdir -p ~/.tabby
cp tabby_config_example.toml ~/.tabby/config.toml
```

Or manually create/edit `~/.tabby/config.toml` with:

```toml
[model.completion.http]
kind = "openai/completion"
api_endpoint = "http://localhost:8741/v1"
model_name = "sweepai/sweep-next-edit-v2-7B"
```

2. In the IDE, go to **Settings** > **Tools** > **Tabby**.
3. Set the **API Endpoint** to `http://localhost:8741`.
4. Leave the API token field empty (newsweep doesn't require authentication).
5. Click **Apply**.

### Step 3: Verify

1. Make sure the newsweep server is running (`./start.sh` or via launchd).
2. Open a source file in the IDE and start typing -- you should see completions appear.
3. Check the server stats at `http://localhost:8741/v1/stats` to confirm requests are being processed.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/completions` | POST | Tabby-compatible completion endpoint |
| `/v1/health` | GET | Server health and model info |
| `/v1/stats` | GET | Latency and throughput metrics |
| `/v1/models` | GET | List loaded models |

## Configuration

Edit `sweep_local/config.py` to adjust:

- `NUM_LINES_BEFORE` / `NUM_LINES_AFTER` -- code block size around cursor
- `MAX_NEW_TOKENS` -- max tokens per completion
- `NUM_DRAFT_TOKENS` -- speculative decoding aggressiveness
- `PROJECT_ROOT` -- directory to watch for file changes

## Performance

Typical performance on Apple Silicon (M2 Pro, 16 GB):

| Metric | Value |
|---|---|
| First token latency | ~200-400 ms |
| Generation speed | ~30-50 tokens/sec |
| Memory usage | ~6-8 GB unified memory |
| Cache hit rate | ~60-80% (consecutive edits) |

Performance varies with model size, prompt length, and chip. Check real-time stats at `http://localhost:8741/v1/stats`.

## Troubleshooting

**Server won't start**
- Ensure models are downloaded and placed in `models/sweep-next-edit-v2-7B-4bit` and `models/qwen2.5-0.5b-4bit`.
- Check that Python 3.12+ is installed: `python --version`.
- Check logs in `logs/stderr.log`.

**No completions in the IDE**
- Verify the server is running: `curl http://localhost:8741/v1/health`.
- Check `~/.tabby/config.toml` points to `http://localhost:8741/v1`.
- In JetBrains settings, make sure the Tabby endpoint is `http://localhost:8741`.
- Restart the IDE after changing Tabby settings.

**Slow completions**
- Close memory-heavy applications to free unified memory.
- Reduce `MAX_NEW_TOKENS` in `sweep_local/config.py` for shorter but faster completions.
- Check `http://localhost:8741/v1/stats` for latency percentiles.

**High memory usage**
- The 7B model (4-bit) uses ~4 GB plus ~1 GB for the draft model. This is expected.
- If memory is tight, you can disable speculative decoding by setting `DRAFT_MODEL_PATH = ""` in `sweep_local/config.py`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

For bugs or feature requests, please [open an issue](https://github.com/raphaelfh/newsweep/issues).

## Acknowledgments

- [SweepAI](https://github.com/sweepai) for the [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B) model
- [MLX](https://github.com/ml-explore/mlx) team at Apple for the ML framework and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [TabbyML](https://github.com/TabbyML/tabby) for the IDE plugin and completion protocol
- [Qwen](https://github.com/QwenLM/Qwen2.5) team for the Qwen2.5-0.5B draft model
