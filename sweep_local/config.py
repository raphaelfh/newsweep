from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = str(_BASE / "sweep-next-edit-v2-7B-4bit")
DRAFT_MODEL_PATH = str(_BASE / "qwen2.5-0.5b-4bit")
MODEL_ID = "sweepai/sweep-next-edit-v2-7B"  # display name
DEVICE = "mlx"

HOST = "0.0.0.0"
PORT = 8741

# Prompt construction
NUM_LINES_BEFORE = 12    # code block lines before cursor
NUM_LINES_AFTER = 12     # code block lines after cursor
MAX_NEW_TOKENS = 128     # allow longer completions for block/method scenarios

# Speculative decoding
NUM_DRAFT_TOKENS = 2     # tokens drafted per step (lower = less waste on rejection)
USE_NGRAM_SPECULATION = True  # use n-gram lookup instead of draft model
NGRAM_N = 3              # n-gram size for lookup (match last N-1 tokens)

# Repetition penalty
REPETITION_PENALTY = 1.2        # penalise already-generated tokens (1.0 = off)
REPETITION_CONTEXT_SIZE = 64    # how many past tokens the penalty looks at
CYCLE_DETECT_WINDOW = 12        # force-stop when last 2*W tokens are a repeated cycle

# Token healing
ENABLE_TOKEN_HEALING = True
MAX_HEALING_TOKENS = 3  # max tokens to roll back for healing

# Early cancellation
EARLY_STOP_MATCH_TOKENS = 4  # consecutive suffix-matching tokens to trigger early stop

# File watcher
PROJECT_ROOT = Path.home() / "PycharmProjects"
MAX_DIFFS_PER_FILE = 10  # rolling buffer of recent diffs
MAX_CROSS_FILE_DIFFS = 3  # max diffs from other files to include as context

# PSI (Program Structure Interface) context
MAX_PSI_TOKENS = 1500    # max tokens allocated to PSI definitions
MAX_PSI_DEFINITIONS = 10 # max definitions per request
