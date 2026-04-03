"""
Prompt construction for sweep-next-edit-v2-7B.
Adapted from https://huggingface.co/sweepai/sweep-next-edit-v2-7B/blob/main/inference.py
"""

from dataclasses import dataclass

PROMPT_TEMPLATE = """<|file_sep|>{file_path}
{initial_file}{retrieval_results}
{recent_changes}
<|file_sep|>original/{file_path}:{start_line}:{end_line}
{prev_section}
<|file_sep|>current/{file_path}:{start_line}:{end_line}
{code_block}
<|file_sep|>updated/{file_path}:{start_line}:{end_line}
{prefill}"""

DIFF_FORMAT = """<|file_sep|>{file_path}:{start_line}:{end_line}
original:
{old_code}
updated:
{new_code}"""

STOP_TOKENS = [
    "<|endoftext|>",
    "<|file_sep|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|im_start|>",
    "<|im_end|>",
]

# Context window around cursor (lines above + below)
NUM_CONTEXT_LINES_HALF = 50

# Snap context window to chunks of this size so small cursor
# movements don't shift the window and invalidate the KV cache.
CONTEXT_CHUNK_SIZE = 40


@dataclass
class FileChunk:
    file_path: str
    content: str

    def to_string(self) -> str:
        return f"<|file_sep|>{self.file_path}\n{self.content}\n"


def compute_prefill(
    code_block: str,
    relative_cursor: int,
    changes_above_cursor: bool = False,
) -> str:
    """
    Compute the prefill — unchanged lines we feed so the model only generates the edit.

    changes_above_cursor=True (insertion mode):
        Only prefill first line + trailing blank lines. Model rewrites from line 2.
    changes_above_cursor=False (default):
        Prefill everything up to the cursor line. Model edits at/below cursor.
    """
    if changes_above_cursor:
        prefill = code_block[:relative_cursor]
        prefilled_lines = prefill.splitlines(True)
        NUM_LINES_ABOVE = 1
        before_split = "".join(prefilled_lines[:NUM_LINES_ABOVE])
        after_split = "".join(prefilled_lines[NUM_LINES_ABOVE:])
        leading_newlines = len(after_split) - len(after_split.lstrip("\n"))
        before_split += "\n" * leading_newlines
        return before_split
    else:
        prefix_before_cursor = code_block[:relative_cursor]
        if "\n" not in prefix_before_cursor:
            return ""
        prefill_end = prefix_before_cursor.rfind("\n") + 1
        return code_block[:prefill_end]


def is_pure_insertion_above_cursor(
    code_block: str, completion: str, relative_cursor: int
) -> bool:
    """
    Reject completions that only insert above cursor without editing the cursor line.
    """
    if not code_block or relative_cursor <= 0:
        return False
    current_line_index = len(code_block[:relative_cursor].splitlines(True))
    code_block_lines = code_block.splitlines(True)
    if current_line_index < 1 or current_line_index > len(code_block_lines):
        return False
    cursor_line = code_block_lines[current_line_index - 1]

    if code_block.strip() == completion.strip():
        return False
    if not cursor_line.strip():
        return False

    prefix_lines = code_block_lines[:current_line_index - 1]
    prefix = "".join(prefix_lines)
    suffix_lines = code_block_lines[current_line_index:]
    suffix = "".join(suffix_lines)

    return completion.startswith(prefix) and completion.endswith(cursor_line + suffix)


def build_prompt(
    file_path: str,
    file_contents: str,
    cursor_position: int,
    recent_changes: str = "",
    retrieval_chunks: list[FileChunk] | None = None,
    file_chunks: list[FileChunk] | None = None,
    changes_above_cursor: bool = False,
    num_lines_before: int = 10,
    num_lines_after: int = 10,
) -> tuple[str, str, int, int]:
    """
    Build the model prompt from file contents and cursor position.

    Returns:
        (formatted_prompt, code_block, block_start_index, relative_cursor)
    """
    lines = file_contents.splitlines(True)
    if not lines:
        return "", "", 0, 0

    # Find cursor line
    pos = 0
    cursor_line = 0
    for i, line in enumerate(lines):
        if pos + len(line) > cursor_position:
            cursor_line = i
            break
        pos += len(line)
    else:
        cursor_line = len(lines) - 1

    # Extract code block around cursor
    block_start = max(0, cursor_line - num_lines_before)
    block_end = min(len(lines), cursor_line + num_lines_after + 1)
    code_block = "".join(lines[block_start:block_end])
    block_start_index = sum(len(line) for line in lines[:block_start])

    # Relative cursor position within code block
    relative_cursor = cursor_position - block_start_index

    # Insert <|cursor|> marker
    code_block_with_cursor = (
        code_block[:relative_cursor]
        + "<|cursor|>"
        + code_block[relative_cursor:]
    )

    # prev_section = the "original" version the model compares against.
    # When there are no recent_changes (no diff between original and current),
    # truncate prev_section at the cursor line so the model sees a visible
    # diff — the cursor line and everything after it appears "new" in the
    # current section, signalling that the developer is inserting content.
    # This prevents the model from predicting "no edit" for pure insertions.
    if not recent_changes:
        prefill_end = len(compute_prefill(code_block, relative_cursor, changes_above_cursor))
        prev_section = code_block[:prefill_end]
    else:
        prev_section = code_block

    prefill = compute_prefill(code_block, relative_cursor, changes_above_cursor)

    # Broad context around cursor — reduce when PSI retrieval chunks provide
    # semantic definitions, since the broad window is less critical then.
    context_half = NUM_CONTEXT_LINES_HALF
    if retrieval_chunks:
        context_half = max(25, NUM_CONTEXT_LINES_HALF - 25)
    # Snap to CONTEXT_CHUNK_SIZE boundaries so small cursor movements
    # don't shift the window and invalidate the KV cache prefix.
    chunk = CONTEXT_CHUNK_SIZE
    anchor = (cursor_line // chunk) * chunk
    context_start = max(0, anchor - context_half)
    context_end = min(len(lines), anchor + chunk + context_half)
    context_before = "".join(lines[context_start:block_start])
    context_after = "".join(lines[block_end:context_end])
    initial_file = context_before + context_after

    retrieval_results = ""
    if retrieval_chunks:
        retrieval_results = "".join(
            f"\n{chunk.to_string()}" for chunk in retrieval_chunks
        )

    start_line = block_start + 1
    end_line = block_end

    formatted = PROMPT_TEMPLATE.format(
        file_path=file_path,
        initial_file=initial_file,
        retrieval_results=retrieval_results,
        recent_changes=recent_changes,
        prev_section=prev_section,
        code_block=code_block_with_cursor,
        start_line=start_line,
        end_line=end_line,
        prefill=prefill,
    )

    if file_chunks:
        formatted = "".join(c.to_string() for c in file_chunks) + formatted

    return formatted, code_block, block_start_index, relative_cursor


def format_diff(
    file_path: str,
    start_line: int,
    end_line: int,
    old_code: str,
    new_code: str,
) -> str:
    """Format a single diff entry for the recent_changes section."""
    return DIFF_FORMAT.format(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        old_code=old_code,
        new_code=new_code,
    )

