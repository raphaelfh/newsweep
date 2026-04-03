# Early Cancellation

Early cancellation detects when the model is reproducing code that already exists after the cursor and stops generation, appending the remaining known text directly.

## The problem

In a next-edit prediction task, the model rewrites a code block around the cursor. Often, the edit only changes a few lines near the cursor, and the rest of the block remains identical. Without early cancellation, the model would spend time generating tokens for unchanged lines:

```python
# Code block (cursor at line 3):
def compute(values):          # line 1 - unchanged
    total = 0                 # line 2 - unchanged
    for v in values:          # line 3 - EDIT: was "for v in vals:"
        total += v            # line 4 - unchanged
    return total / len(values)# line 5 - unchanged
```

After the model edits line 3, it starts reproducing lines 4-5 token by token. Those tokens are predictable -- they match the suffix (code after the cursor).

## How it works

### Setup

Before generation starts, the server tokenizes the **suffix** -- the text in the code block from the cursor position to the end:

```
suffix_text = code_block[relative_cursor:]
suffix_tokens = tokenizer.encode(suffix_text)
```

### Detection

During generation, each produced token is compared against the expected suffix token:

```
Generated:  [edit_tok1, edit_tok2, suffix_tok0, suffix_tok1, ..., suffix_tok9]
Expected:                         [suffix_tok0, suffix_tok1, ..., suffix_tok9, ...]
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   10 consecutive matches -> early stop!
```

The comparison tracks two values:
- `suffix_pos`: Current position in the suffix token list
- `suffix_match_count`: Consecutive matches so far

When a generated token matches `suffix_tokens[suffix_pos]`, the counter increments. On mismatch, both reset to zero.

### Trigger

After `EARLY_STOP_MATCH_TOKENS` (default: 4) consecutive matches, the server:

1. Stops generation
2. Appends the remaining suffix tokens directly to the output
3. Decodes the full token list

This works because if the model has reproduced 10 consecutive suffix tokens, it's almost certainly reproducing the rest verbatim.

### Cache invalidation

After early stopping, the KV cache is **not** saved for the next request. The appended suffix tokens were never processed by the model, so the cache would contain incorrect entries. This is a small cost -- the next request starts with a fresh cache lookup.

## Example

```
Code block (20 lines):
  Lines 1-8:   context above cursor
  Line 9:      cursor line (being edited)
  Lines 10-20: context below cursor (11 lines = the suffix)

Without early stop:
  Model generates all 20 lines: ~50 tokens, ~500ms

With early stop:
  Model generates lines 1-9 (the edit): ~25 tokens
  Model starts reproducing line 10: matches suffix
  After 10 matching tokens: STOP
  Append remaining suffix tokens
  Total generation: ~30 tokens, ~300ms + instant suffix append
```

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `EARLY_STOP_MATCH_TOKENS` | `4` | Consecutive matches to trigger early stop |

**Lower values** (e.g., 2-3): Faster but riskier. The model might be generating similar-but-not-identical code, and stopping early would produce incorrect output.

**Higher values** (e.g., 10-20): Safer but less speedup. More tokens must be generated before the optimization kicks in.

The default of 4 balances speed and accuracy. At this threshold, false positives are extremely rare while maximizing early termination speed.

## Interaction with speculation

Early cancellation works with all three generation paths (n-gram, draft model, standard). During n-gram speculation, both the base token and accepted draft tokens are checked against the suffix. If the early stop triggers during draft verification, unverified draft tokens are trimmed from the cache.
