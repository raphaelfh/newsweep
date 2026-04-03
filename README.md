# newsweep

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black.svg)](https://support.apple.com/en-us/116943)

Local next-edit autocomplete server for JetBrains IDEs via [Tabby](https://tabby.tabbyml.com/), powered by [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B) running on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## What it does

newsweep runs a local inference server that predicts your *next edit* rather than just completing the current line. It watches your project files for changes, uses recent diffs as context, and applies several optimizations from [Sweep's engineering blog](https://blog.sweep.dev/) to deliver fast, accurate suggestions.

Features:

- **Next-edit prediction** -- suggests edits based on your recent changes, not just cursor position
- **N-gram speculative decoding** -- ~5x faster than baseline by exploiting the fact that >90% of tokens are unchanged during rewrites
- **Token healing** -- fixes accuracy problems caused by partial-word tokenization (sub-1ms via vocabulary trie)
- **Early cancellation** -- stops generation as soon as the edit is complete, skipping redundant trailing tokens
- **Cross-file context** -- includes recent diffs from other files to capture multi-file edit patterns
- **PSI context** -- accepts type-resolved definitions from JetBrains for richer completions
- **KV cache reuse** -- reuses computation across consecutive requests for lower latency
- **Request cancellation** -- fast typing cancels in-flight requests so the server stays responsive
- **Tabby-compatible API** -- works with the Tabby plugin for PyCharm (and other JetBrains IDEs)

## How it works

```
 JetBrains IDE (Tabby plugin)
        |
        |  POST /v1/completions  {file_content, cursor_position, psi_context}
        v
 +-------------------------------------------------------------------+
 |  newsweep server (FastAPI on Apple Silicon)                        |
 |                                                                    |
 |  1. File Watcher ----> tracks diffs (same-file + cross-file)      |
 |  2. Token Healing ---> fixes partial-word tokenization boundaries  |
 |  3. Prompt Builder --> code block + diffs + PSI definitions        |
 |  4. MLX Inference                                                  |
 |     - N-gram speculative decoding (default)                       |
 |     - Early cancellation on suffix match                          |
 |     - KV cache reuse between requests                             |
 +-------------------------------------------------------------------+
```

The server watches your project directory for file saves. When you edit code, recent diffs are recorded and injected into the model prompt as context -- both from the current file and from other recently modified files. This allows the model to predict what you're likely to edit *next*, capturing multi-file patterns like "update a function signature, then update all callers."

## Key techniques

### N-gram speculative decoding

Standard autoregressive generation produces one token per forward pass. Speculative decoding speeds this up by proposing multiple tokens, then verifying them in a single batch forward pass.

The insight from [Sweep's blog](https://blog.sweep.dev/posts/next-edit-jetbrains) is that **>90% of tokens remain unchanged** when the model rewrites code around the cursor. Instead of using a separate draft model (which requires 500MB of extra memory and still only achieves ~1.2x speedup), we search the prompt itself for n-gram matches:

1. Build a hash index of all n-grams in the prompt tokens
2. After generating each token, look up the last few tokens in the index
3. If a match is found, copy the following N tokens as draft proposals
4. Feed all draft tokens through the main model in one forward pass
5. Accept matching tokens, reject on first mismatch

Because the model is mostly reproducing unchanged code, the acceptance rate is high, yielding ~5x total speedup over standard generation -- without any extra model.

### Token healing

LLMs are trained on complete text, but during autocomplete the code is in a partial state. When a developer types `Nod` (intending `Node`), the tokenizer splits this into unusual token boundaries (`N` + `od`) that the model has never seen in training. This causes wrong completions.

Token healing, based on [Sweep's approach](https://blog.sweep.dev/posts/token-healing-autocomplete), fixes this at inference time:

1. **Detect unstable boundaries** -- walk backward from the end of the token sequence, re-encoding to find where tokenization becomes unstable
2. **Roll back** -- remove the unstable tokens and save the corresponding text as a "healing prefix"
3. **Constrain generation** -- use a logits mask to force the model to only produce tokens that either extend the healing prefix or consume part of it
4. **Trie-based lookup** -- a vocabulary trie enables sub-1ms prefix matching over 150K tokens, with pre-computed caches for high-frequency prefixes (like space, which covers 50K+ tokens)

The result: the model sees clean token boundaries and is constrained to produce output consistent with what the developer has already typed.

### Early cancellation

Next-edit completions rewrite a block of code around the cursor. Often the actual edit is small (a few tokens changed), but the model continues generating the unchanged suffix. Early cancellation detects when the model's output re-aligns with the known suffix of the code block:

- Tokenize the code after the cursor as a reference suffix
- During generation, track consecutive matches against this suffix
- After N consecutive matches (default: 10), stop generation and append the remaining known suffix

This avoids wasting compute on tokens we already know, providing up to ~4x decoding speedup on typical edits.

### Cross-file context

When developers make related changes across multiple files (rename a function, update its callers), the model benefits from seeing those other changes. The file watcher maintains a rolling buffer of recent diffs from all project files, and the prompt builder includes the most recent cross-file diffs alongside same-file diffs.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- ~4 GB of disk space for the main model
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

3. Download and convert the main model.

   newsweep requires one model converted to MLX 4-bit format. The `mlx_lm` CLI (installed with the project dependencies in step 2) handles downloading from Hugging Face and quantizing in one command.

   **Main model** -- [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B), the next-edit prediction model by SweepAI:

   ```bash
   mlx_lm.convert \
     --hf-path sweepai/sweep-next-edit-v2-7B \
     -q --q-bits 4 \
     --mlx-path models/sweep-next-edit-v2-7B-4bit
   ```

   This downloads the original model weights (~14 GB), quantizes them to 4-bit (~4 GB), and saves the result in MLX format.

   **Draft model (optional)** -- only needed if you disable n-gram speculation (`USE_NGRAM_SPECULATION = False` in `config.py`). The default n-gram approach is faster and doesn't require a second model.

   ```bash
   mlx_lm.convert \
     --hf-path Qwen/Qwen2.5-0.5B \
     -q --q-bits 4 \
     --mlx-path models/qwen2.5-0.5b-4bit
   ```

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
| `/v1/health` | GET | Server health, model info, and enabled features |
| `/v1/stats` | GET | Latency, throughput, cache, healing, and draft metrics |
| `/v1/models` | GET | List loaded models |

## Configuration

All settings are in `sweep_local/config.py`:

| Setting | Default | Description |
|---|---|---|
| **Inference** | | |
| `NUM_LINES_BEFORE` | `10` | Code block lines above cursor |
| `NUM_LINES_AFTER` | `10` | Code block lines below cursor |
| `MAX_NEW_TOKENS` | `1024` | Max tokens per completion |
| **Speculative decoding** | | |
| `USE_NGRAM_SPECULATION` | `True` | Use n-gram lookup instead of draft model |
| `NGRAM_N` | `4` | N-gram size for lookup (match last N-1 tokens) |
| `NUM_DRAFT_TOKENS` | `3` | Tokens drafted per speculative step |
| **Token healing** | | |
| `ENABLE_TOKEN_HEALING` | `True` | Fix partial-word tokenization boundaries |
| `MAX_HEALING_TOKENS` | `3` | Max tokens to roll back for healing |
| **Early cancellation** | | |
| `EARLY_STOP_MATCH_TOKENS` | `10` | Consecutive suffix matches to trigger early stop |
| **Context** | | |
| `PROJECT_ROOT` | `~/PycharmProjects` | Directory to watch for file changes |
| `MAX_DIFFS_PER_FILE` | `10` | Rolling buffer of recent diffs per file |
| `MAX_CROSS_FILE_DIFFS` | `3` | Max diffs from other files in prompt |
| `MAX_PSI_TOKENS` | `1500` | Token budget for PSI definitions |
| `MAX_PSI_DEFINITIONS` | `10` | Max type definitions per request |

## Performance

Typical performance on Apple Silicon (M2 Pro, 16 GB) with n-gram speculation and token healing enabled:

| Metric | Value |
|---|---|
| First token latency | ~200-400 ms |
| Generation speed | ~30-50 tokens/sec |
| N-gram draft acceptance | ~60-90% (code rewrites) |
| Memory usage | ~6 GB unified memory |
| Cache hit rate | ~60-80% (consecutive edits) |
| Token healing overhead | <1 ms |

Performance varies with model size, prompt length, and chip. Check real-time stats at `http://localhost:8741/v1/stats`.

## Troubleshooting

**Server won't start**
- Ensure the main model is downloaded to `models/sweep-next-edit-v2-7B-4bit/`.
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
- Ensure `USE_NGRAM_SPECULATION = True` (default) for fastest generation.
- Check `http://localhost:8741/v1/stats` for latency percentiles and draft acceptance rate.

**Wrong completions on partial words**
- Ensure `ENABLE_TOKEN_HEALING = True` (default).
- Check `/v1/stats` -- the `healing_applied` counter should be non-zero.

**High memory usage**
- The 7B model (4-bit) uses ~4 GB. With n-gram speculation (default), no draft model is loaded.
- If using draft-model speculation instead, add ~1 GB for the Qwen2.5-0.5B model.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

For bugs or feature requests, please [open an issue](https://github.com/raphaelfh/newsweep/issues).

## Acknowledgments

- [SweepAI](https://github.com/sweepai) for the [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B) model and the engineering blog posts on [speculative decoding](https://blog.sweep.dev/posts/next-edit-jetbrains) and [token healing](https://blog.sweep.dev/posts/token-healing-autocomplete) that inspired the optimizations in this project
- [MLX](https://github.com/ml-explore/mlx) team at Apple for the ML framework and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [TabbyML](https://github.com/TabbyML/tabby) for the IDE plugin and completion protocol
- [Qwen](https://github.com/QwenLM/Qwen2.5) team for the Qwen2.5-0.5B draft model
