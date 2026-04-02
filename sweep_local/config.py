from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = str(_BASE / "sweep-next-edit-v2-7B-4bit")
DRAFT_MODEL_PATH = str(_BASE / "qwen2.5-0.5b-4bit")
MODEL_ID = "sweepai/sweep-next-edit-v2-7B"  # display name
DEVICE = "mlx"

HOST = "0.0.0.0"
PORT = 8741

# Prompt construction
NUM_LINES_BEFORE = 10    # code block lines before cursor
NUM_LINES_AFTER = 10     # code block lines after cursor
MAX_NEW_TOKENS = 128

# Speculative decoding
NUM_DRAFT_TOKENS = 3     # tokens drafted per step (higher = more speculative)

# File watcher
PROJECT_ROOT = Path.home() / "PycharmProjects"
MAX_DIFFS_PER_FILE = 10  # rolling buffer of recent diffs
