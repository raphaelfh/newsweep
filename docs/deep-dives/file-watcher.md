# File Watcher

The file watcher monitors your project directory for file saves and records diffs. These diffs are injected into the model prompt as "recent changes" context, enabling the model to predict your *next* edit based on what you've been doing.

## Why diffs matter

The sweep-next-edit-v2-7B model is trained to predict edits, not just complete code. By showing it your recent changes, it can identify patterns:

- If you renamed a variable in one function, it might suggest the same rename in the next function
- If you added error handling to one endpoint, it might suggest similar handling for another
- If you updated a function signature, it might suggest updating the callers

This is what makes newsweep a "next-edit" predictor rather than a generic code completer.

## Architecture

```
PROJECT_ROOT (e.g., ~/PycharmProjects)
    |
    |  file save events
    v
watchdog Observer (daemon thread)
    |
    |  on_modified events
    v
FileChangeHandler
    |
    |  reads new file content
    |  computes diff against cached version
    v
DiffStore (thread-safe, global)
    |
    |  rolling buffer per file
    v
server.py build_recent_changes()
    |
    v
prompt "recent_changes" section
```

## Components

### FileChangeHandler

A `watchdog.FileSystemEventHandler` subclass that listens for `FileModifiedEvent`s. When a file is saved:

1. Check if the file extension is in the watched set
2. Read the new file content
3. Pass it to `diff_store.update_file()`

Watched extensions cover most programming languages: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.kt`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp`, `.rb`, `.php`, `.swift`, `.scala`, `.cs`, `.vue`, `.svelte`, `.html`, `.css`, `.scss`, `.json`, `.yaml`, `.yml`, `.toml`, `.md`, `.txt`, `.sh`.

### DiffStore

A thread-safe store of recent diffs per file. Key behaviors:

**`update_file(file_path, new_content)`**
- Compares new content against the cached version using `difflib.SequenceMatcher`
- Records a `FileDiff` for each changed block (insert, delete, or replace)
- Updates the file cache with the new content
- Returns the list of new diffs
- First call for a file: caches content but returns no diffs (no previous version to compare)

**`get_recent_diffs(file_path=None)`**
- With a file path: returns diffs for that file only
- Without a file path: returns all diffs across all files, sorted by timestamp

**`seed_file(file_path, content)`**
- Updates the cache without generating diffs
- Used by the server when the IDE sends the current file content with a completion request

Each file has a rolling buffer of `MAX_DIFFS_PER_FILE` (default: 10) diffs. Older diffs are discarded.

### FileDiff

```python
@dataclass
class FileDiff:
    file_path: str      # which file changed
    old_chunk: str       # the original text
    new_chunk: str       # the replacement text
    start_line: int      # 1-based line number
    end_line: int        # 1-based line number
    timestamp: float     # time.time()
```

## How diffs enter the prompt

When the server handles a completion request, `build_recent_changes()` gathers diffs from two sources:

### Same-file diffs (up to 3)

The most recent 3 diffs for the current file. These show the model what the user has been changing in the file being edited.

### Cross-file diffs (up to `MAX_CROSS_FILE_DIFFS`)

The most recent diffs from *other* files. These capture multi-file edit patterns -- for example, if you updated an API endpoint definition in one file, the model might suggest updating the corresponding client code in another file.

Each diff is formatted as:

```
<|file_sep|>path/to/file.py:10:15
original:
    old_value = compute(x)
updated:
    new_value = compute(x, y)
```

## Configuration

| Setting | Default | Description |
|---|---|---|
| `PROJECT_ROOT` | `~/PycharmProjects` | Root directory to watch |
| `MAX_DIFFS_PER_FILE` | `10` | Rolling buffer size per file |
| `MAX_CROSS_FILE_DIFFS` | `3` | Max diffs from other files in prompt |

## Edge cases

- **Binary files**: Ignored (only watched extensions are processed)
- **Permission errors**: Silently skipped
- **Encoding**: Files are read with `utf-8` encoding and `errors="replace"`
- **Rapid saves**: Each save produces a diff against the last cached version, so rapid saves create multiple diffs
- **IDE content seeding**: The server calls `seed_file()` on each request to keep the cache synchronized with the IDE's view of the file, even if the file hasn't been saved to disk yet
