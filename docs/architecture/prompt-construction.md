# Prompt Construction

The sweep-next-edit-v2-7B model expects a specific prompt format. The `sweep_prompt.py` module assembles this format from raw file contents, cursor position, diffs, and optional retrieval context.

## Prompt template

```
<|file_sep|>path/to/file.py
{broad context: ~150 lines around cursor, excluding code block}
{retrieval chunks: PSI definitions, cross-file context}
{recent changes: formatted diffs from file watcher}
<|file_sep|>original/path/to/file.py:11:21
{code block without cursor marker - the "before" version}
<|file_sep|>current/path/to/file.py:11:21
{code block with <|cursor|> marker - what the IDE shows}
<|file_sep|>updated/path/to/file.py:11:21
{prefill - unchanged lines the model should reproduce}
```

The model generates tokens after the prefill. It outputs an updated version of the code block, and the server extracts the edit.

## Sections explained

### 1. Broad context (initial_file)

The first section gives the model file-level awareness. It includes ~150 lines above and ~150 lines below the cursor, **excluding** the code block itself (to avoid duplication).

When PSI retrieval chunks are present, this window shrinks to ~100 lines per side. The PSI definitions provide semantic context (type signatures, method bodies), so less raw code context is needed.

### 2. Retrieval results

Optional cross-file context appended after the initial file. Each chunk is wrapped in `<|file_sep|>` tags with the source file path. Currently used for:

- **PSI definitions**: Type-resolved symbols from the JetBrains IDE (classes, methods, fields). Sorted by priority, trimmed to a token budget (`MAX_PSI_TOKENS`).
- **File chunks**: Any additional context (e.g., related files).

### 3. Recent changes

Formatted diffs from the file watcher, showing what the user has been editing recently. This is the key signal for next-edit prediction -- the model sees what you changed and predicts what you'll change next.

Each diff is formatted as:

```
<|file_sep|>path/to/file.py:10:15
original:
{old code}
updated:
{new code}
```

The server includes up to 3 same-file diffs and up to `MAX_CROSS_FILE_DIFFS` (default 3) from other recently modified files.

### 4. Original section (prev_section)

The code block *without* the cursor marker. This shows the model the "original" state of the code around the cursor.

### 5. Current section (code_block with cursor)

The same code block, but with a `<|cursor|>` marker inserted at the exact cursor position. This tells the model where the user is editing.

### 6. Prefill

Unchanged lines that the model should reproduce verbatim. This avoids wasting generation capacity on text that won't change.

**Default mode** (`changes_above_cursor=False`): Prefill includes all lines up to (but not including) the cursor line. The model starts generating from the cursor line onward.

**Insertion mode** (`changes_above_cursor=True`): Only the first line and any trailing blank lines are prefilled. The model can insert new lines starting from line 2.

## Code block extraction

`build_prompt()` extracts a window of `NUM_LINES_BEFORE` + `NUM_LINES_AFTER` lines centered on the cursor:

```
line 1    |
line 2    |  (broad context above)
...       |
line 10   |
line 11   <- block_start (cursor_line - NUM_LINES_BEFORE)
line 12   |
...       |  (code block: 21 lines total)
line 21   <- cursor_line (with <|cursor|> marker)
...       |
line 31   <- block_end (cursor_line + NUM_LINES_AFTER)
line 32   |
...       |  (broad context below)
```

The function returns four values:

| Return value | Type | Description |
|---|---|---|
| `formatted_prompt` | `str` | Complete prompt string ready for tokenization |
| `code_block` | `str` | The extracted code block (without cursor marker) |
| `block_start_index` | `int` | Character offset of the code block within the file |
| `relative_cursor` | `int` | Cursor position relative to the code block start |

## Stop tokens

Generation stops when the model produces any of these tokens:

```
<|endoftext|>  <|file_sep|>  <|fim_prefix|>  <|fim_middle|>
<|fim_suffix|>  <|fim_pad|>  <|repo_name|>  <|im_start|>  <|im_end|>
```

Plus the tokenizer's EOS token.

## Pure insertion rejection

After generation, the server checks if the completion only inserted text *above* the cursor line without actually editing the cursor line. These completions are rejected (returned as empty) because they don't represent a meaningful edit at the cursor.

Exception: if the cursor is on a blank line, insertions above are allowed, since the user may be adding new code between existing blocks.

## Example

Given this file with the cursor at `|`:

```python
import os

def main():
    path = os.getcwd()
    files = os.listdir(|path)
    return files
```

The prompt would include:
1. **Broad context**: The full file (small file, fits in the window)
2. **Recent changes**: Any recent diffs (e.g., if you just added the `import os` line)
3. **Original**: The code block around the cursor, without marker
4. **Current**: The code block with `<|cursor|>` between `(` and `path`
5. **Prefill**: Lines before the cursor line (`import os\n\ndef main():\n    path = os.getcwd()\n`)

The model then generates the updated code block, potentially completing or modifying the `os.listdir()` call.
