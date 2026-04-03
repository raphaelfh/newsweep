# newsweep

Local next-edit autocomplete server for PyCharm via [Tabby](https://tabby.tabbyml.com/), powered by [sweep-next-edit-v2-7B](https://huggingface.co/sweepai/sweep-next-edit-v2-7B) running on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## What it does

newsweep runs a local inference server that provides intelligent code completions by predicting your *next edit* rather than just completing the current line. It watches your project files for changes and uses recent diffs as context, so suggestions are aware of what you've been working on.

Key features:

- **Next-edit prediction** -- suggests edits based on your recent changes, not just cursor position
- **Speculative decoding** -- uses a small Qwen2.5-0.5B draft model for ~1.2x faster generation
- **KV cache reuse** -- reuses computation across consecutive requests for lower latency
- **Request cancellation** -- fast typing cancels in-flight requests so the server stays responsive
- **Tabby-compatible API** -- works with the Tabby plugin for PyCharm (and other IDEs)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+

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

3. Download the models into the `models/` directory:

```bash
mkdir -p models
# Download sweep-next-edit-v2-7B (MLX 4-bit) and qwen2.5-0.5b (MLX 4-bit)
# Place them in models/sweep-next-edit-v2-7B-4bit and models/qwen2.5-0.5b-4bit
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
| `/v1/health` | GET | Server health and model info |
| `/v1/stats` | GET | Latency and throughput metrics |
| `/v1/models` | GET | List loaded models |

## Configuration

Edit `sweep_local/config.py` to adjust:

- `NUM_LINES_BEFORE` / `NUM_LINES_AFTER` -- code block size around cursor
- `MAX_NEW_TOKENS` -- max tokens per completion
- `NUM_DRAFT_TOKENS` -- speculative decoding aggressiveness
- `PROJECT_ROOT` -- directory to watch for file changes

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository and create a feature branch from `main`.
2. **Install** development dependencies:
   ```bash
   pip install -e .
   ```
3. **Make your changes** -- keep commits focused and well-described.
4. **Test** your changes locally by running the server and verifying completions work.
5. **Open a pull request** against `main` with a clear description of what you changed and why.

### Guidelines

- Follow existing code style and conventions.
- Keep PRs small and focused on a single change.
- Add or update docstrings for any new public functions.
- If adding a new feature, update this README as needed.
- Be respectful in discussions and code reviews.

For bugs or feature requests, please [open an issue](https://github.com/raphaelfh/newsweep/issues).
