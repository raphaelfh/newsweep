# Data Model: Fix Empty Completions

**Feature**: 001-fix-empty-completions
**Date**: 2026-04-03

## Entities

This is a bug fix — no new entities are introduced. The existing entities involved in the completion pipeline are documented here for reference.

### CompletionRequest (existing, unchanged)

Received from the IDE client via `POST /v1/completions`.

| Field | Type | Description |
|-------|------|-------------|
| segments.prefix | string | Code before the cursor |
| segments.suffix | string | Code after the cursor |
| segments.filepath | string | Path of the file being edited |
| file_content | string | Full file content |
| cursor_position | int | Character offset of cursor in file |
| max_tokens | int | Maximum tokens to generate (default 128) |

### PromptContext (internal, modified)

Built by `build_prompt()` and fed to the model.

| Field | Type | Description |
|-------|------|-------------|
| prompt_text | string | Full prompt string with `<|file_sep|>` markers |
| code_block | string | The code section containing the cursor |
| relative_cursor | int | Cursor offset within code_block |
| prefill | string | Pre-filled lines the model should continue from |
| start_line | int | First line number of the code block |
| end_line | int | Last line number of the code block |

**Fix impact**: The `prefill` computation and `<|file_sep|>` marker placement in the prompt template may need adjustment to prevent the model from predicting a stop token as the first output token.

### GenerationResult (internal, modified)

Produced by the `generate()` function.

| Field | Type | Description |
|-------|------|-------------|
| generated | list[int] | Token IDs produced by the model |
| completion | string | Decoded text after trimming and filtering |
| hit_stop | bool | Whether generation ended on a stop token |
| cycle_stopped | bool | Whether cycle detection triggered |
| cancelled | bool | Whether request was superseded |

**Fix impact**: Cycle trimming must ensure `generated` is never trimmed to empty. Error logging must capture the state when exceptions occur.

### CompletionResponse (existing, unchanged)

Returned to the IDE client. Tabby-compatible API contract — no changes.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique response ID |
| choices[0].text | string | The completion text |
| choices[0].index | int | Always 0 |

## State Transitions

```
Request received
    → Prompt constructed (build_prompt)
    → Generation started
        → Token sampled
            → [stop token on first sample] → empty generated list → FIX: investigate prompt
            → [stop token after N tokens] → generated has tokens → normal flow
            → [cycle detected] → trimmed generated → FIX: preserve minimum tokens
            → [max tokens reached] → full generated list → normal flow
            → [exception thrown] → FIX: log with context, return ""
    → Post-processing
        → Decode tokens to text
        → Apply STOP_TOKENS text trimming
        → Apply is_pure_insertion_above_cursor filter → FIX: validate + log
    → Response returned
```
