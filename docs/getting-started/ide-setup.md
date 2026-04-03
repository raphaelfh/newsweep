# IDE Setup

newsweep integrates with JetBrains IDEs (PyCharm, IntelliJ, WebStorm, etc.) through the [Tabby](https://tabby.tabbyml.com/) plugin. Tabby is an open-source code completion framework -- newsweep implements its API so the plugin can send completion requests to the local server.

## Step 1: Install the Tabby plugin

1. Open your JetBrains IDE
2. Go to **Settings** > **Plugins** > **Marketplace**
3. Search for **Tabby** and click **Install**
4. Restart the IDE

## Step 2: Configure Tabby

The Tabby plugin needs to know where newsweep is running.

### Option A: Config file (recommended)

Copy the included example config:

```bash
mkdir -p ~/.tabby
cp tabby_config_example.toml ~/.tabby/config.toml
```

Or create `~/.tabby/config.toml` manually:

```toml
[model.completion.http]
kind = "openai/completion"
api_endpoint = "http://localhost:8741/v1"
model_name = "sweepai/sweep-next-edit-v2-7B"
```

### Option B: IDE settings

1. Go to **Settings** > **Tools** > **Tabby**
2. Set **API Endpoint** to `http://localhost:8741`
3. Leave the API token field empty (newsweep doesn't require authentication)
4. Click **Apply**

## Step 3: Verify

1. Make sure the server is running (`./start.sh` or via launchd)
2. Open a source file and start typing -- you should see completions appear
3. Check server stats at `http://localhost:8741/v1/stats` to confirm requests are processed

## How the integration works

When you type in the IDE:

1. The Tabby plugin detects a pause in typing
2. It sends a `POST /v1/completions` request with:
   - `segments.prefix`: code before the cursor
   - `segments.suffix`: code after the cursor
   - `segments.filepath`: path to the current file
   - Optionally: `file_content`, `cursor_position`, and `psi_context` (type-resolved definitions from the IDE's code intelligence)
3. newsweep builds a prompt with the code context, recent diffs, and PSI definitions
4. The model generates a completion
5. The result appears as an inline suggestion in the editor

## Troubleshooting

**No completions appear**
- Verify the server: `curl http://localhost:8741/v1/health`
- Check `~/.tabby/config.toml` points to `http://localhost:8741/v1`
- In IDE settings, confirm the Tabby endpoint is `http://localhost:8741`
- Restart the IDE after changing Tabby settings

**Completions are slow**
- Close memory-heavy applications to free unified memory
- Reduce `MAX_NEW_TOKENS` in `sweep_local/config.py`
- Check latency percentiles at `http://localhost:8741/v1/stats`
