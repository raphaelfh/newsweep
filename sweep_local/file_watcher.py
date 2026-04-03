"""
File watcher that monitors project files for changes and maintains a rolling
buffer of recent diffs per file. These diffs are used as context for the
newsweep model's "recent_changes" prompt section.
"""

import difflib
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from watchdog.observers import Observer

from sweep_local.config import MAX_DIFFS_PER_FILE

@dataclass
class FileDiff:
    file_path: str
    old_chunk: str
    new_chunk: str
    start_line: int
    end_line: int
    timestamp: float = field(default_factory=time.time)


class DiffStore:
    """Thread-safe store of recent diffs per file."""

    def __init__(self, max_diffs: int = MAX_DIFFS_PER_FILE):
        self._diffs: dict[str, deque[FileDiff]] = defaultdict(
            lambda: deque(maxlen=max_diffs)
        )
        self._file_cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def get_cached_content(self, file_path: str) -> str | None:
        with self._lock:
            return self._file_cache.get(file_path)

    def update_file(self, file_path: str, new_content: str) -> list[FileDiff]:
        """Record a file change. Returns diffs for all changed blocks."""
        with self._lock:
            old_content = self._file_cache.get(file_path)
            self._file_cache[file_path] = new_content

            if old_content is None or old_content == new_content:
                return []

            old_lines = old_content.splitlines(True)
            new_lines = new_content.splitlines(True)
            matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

            now = time.time()
            diffs = []
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == "equal":
                    continue
                start_line = i1 + 1
                end_line = max(i2, start_line)
                old_chunk = "".join(old_lines[max(0, start_line - 1):end_line])
                new_chunk = "".join(new_lines[max(0, j1):j2])
                diff = FileDiff(
                    file_path=file_path,
                    old_chunk=old_chunk,
                    new_chunk=new_chunk,
                    start_line=start_line,
                    end_line=end_line,
                    timestamp=now,
                )
                self._diffs[file_path].append(diff)
                diffs.append(diff)
            return diffs

    def get_recent_diffs(self, file_path: str | None = None) -> list[FileDiff]:
        """Get recent diffs, optionally filtered to a specific file."""
        with self._lock:
            if file_path:
                return list(self._diffs.get(file_path, []))
            all_diffs = []
            for diffs in self._diffs.values():
                all_diffs.extend(diffs)
            all_diffs.sort(key=lambda d: d.timestamp)
            return all_diffs

    def seed_file(self, file_path: str, content: str):
        """Seed or update the cache with the IDE's current content (no diff generated)."""
        with self._lock:
            self._file_cache[file_path] = content


# Global diff store
diff_store = DiffStore()

WATCHED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go",
    ".rs", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift",
    ".scala", ".cs", ".vue", ".svelte", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".sh",
}


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and not event.is_directory:
            path = Path(event.src_path)
            if path.suffix in WATCHED_EXTENSIONS:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    diff_store.update_file(str(path), content)
                except (OSError, PermissionError):
                    pass


def start_watcher(project_root: str | Path) -> Observer:
    """Start watching a project directory for file changes. Returns the observer."""
    project_root = Path(project_root)
    observer = Observer()
    handler = FileChangeHandler()
    observer.schedule(handler, str(project_root), recursive=True)
    observer.daemon = True
    observer.start()
    return observer
