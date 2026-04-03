"""Tests for PSI (Program Structure Interface) context integration."""

from sweep_local.server import PsiDefinition, psi_to_chunks
from sweep_local.sweep_prompt import FileChunk, build_prompt


# --- PsiDefinition model ---


def test_psi_definition_minimal():
    d = PsiDefinition(
        kind="class",
        name="Foo",
        qualified_name="pkg.Foo",
        signature="class Foo:",
    )
    assert d.kind == "class"
    assert d.priority == 0
    assert d.file_path is None
    assert d.body_preview is None


def test_psi_definition_full():
    d = PsiDefinition(
        kind="method",
        name="query",
        qualified_name="db.Client.query",
        signature="def query(sql: str) -> list:",
        file_path="db/client.py",
        body_preview="    return self._exec(sql)",
        priority=10,
    )
    assert d.file_path == "db/client.py"
    assert d.priority == 10
    assert d.body_preview is not None


# --- psi_to_chunks ---


def test_psi_to_chunks_basic():
    defs = [
        PsiDefinition(
            kind="class",
            name="Client",
            qualified_name="db.Client",
            signature="class Client:\n    def connect(self): ...\n    def query(self, sql: str): ...",
        ),
    ]
    chunks = psi_to_chunks(defs)
    assert len(chunks) == 1
    assert isinstance(chunks[0], FileChunk)
    assert "class Client:" in chunks[0].content
    assert chunks[0].file_path == "definitions/db.Client"


def test_psi_to_chunks_uses_file_path_when_available():
    defs = [
        PsiDefinition(
            kind="class",
            name="Client",
            qualified_name="db.Client",
            signature="class Client: ...",
            file_path="src/db/client.py",
        ),
    ]
    chunks = psi_to_chunks(defs)
    assert chunks[0].file_path == "src/db/client.py"


def test_psi_to_chunks_includes_body_preview():
    defs = [
        PsiDefinition(
            kind="method",
            name="run",
            qualified_name="app.run",
            signature="def run():",
            body_preview="    start_server()",
        ),
    ]
    chunks = psi_to_chunks(defs)
    assert "start_server()" in chunks[0].content


def test_psi_to_chunks_sorted_by_priority():
    defs = [
        PsiDefinition(kind="class", name="Low", qualified_name="a.Low",
                       signature="class Low:", priority=1),
        PsiDefinition(kind="class", name="High", qualified_name="a.High",
                       signature="class High:", priority=10),
        PsiDefinition(kind="class", name="Mid", qualified_name="a.Mid",
                       signature="class Mid:", priority=5),
    ]
    chunks = psi_to_chunks(defs)
    assert chunks[0].file_path == "definitions/a.High"
    assert chunks[1].file_path == "definitions/a.Mid"
    assert chunks[2].file_path == "definitions/a.Low"


def test_psi_to_chunks_respects_max_definitions():
    defs = [
        PsiDefinition(kind="class", name=f"C{i}", qualified_name=f"pkg.C{i}",
                       signature=f"class C{i}: ...")
        for i in range(20)
    ]
    chunks = psi_to_chunks(defs)
    assert len(chunks) <= 10


def test_psi_to_chunks_respects_token_budget():
    # Each definition ~1500 chars -> ~510 estimated tokens
    # Budget is 1500 tokens, so should fit ~2-3, not all 5
    big_sig = "class Big:\n" + "    def method_xxx(): ...\n" * 50
    defs = [
        PsiDefinition(kind="class", name=f"Big{i}", qualified_name=f"pkg.Big{i}",
                       signature=big_sig, priority=10 - i)
        for i in range(5)
    ]
    chunks = psi_to_chunks(defs)
    assert 1 <= len(chunks) < 5


def test_psi_to_chunks_empty():
    assert psi_to_chunks([]) == []


# --- Prompt integration ---


SAMPLE_FILE = """\
import os

def main():
    client = get_client()
    client.query("SELECT 1")

def get_client():
    return None
"""


def test_build_prompt_with_retrieval_chunks():
    cursor = SAMPLE_FILE.index("client.query")
    chunks = [
        FileChunk(
            file_path="definitions/db.Client",
            content="class Client:\n    def query(self, sql: str) -> list: ...",
        ),
    ]
    prompt, *_ = build_prompt(
        file_path="app.py",
        file_contents=SAMPLE_FILE,
        cursor_position=cursor,
        retrieval_chunks=chunks,
    )
    assert "class Client:" in prompt
    assert "def query" in prompt
    assert "definitions/db.Client" in prompt


def test_build_prompt_without_psi_unchanged():
    cursor = SAMPLE_FILE.index("client.query")
    prompt_no_psi, *_ = build_prompt(
        file_path="app.py",
        file_contents=SAMPLE_FILE,
        cursor_position=cursor,
    )
    prompt_with_psi, *_ = build_prompt(
        file_path="app.py",
        file_contents=SAMPLE_FILE,
        cursor_position=cursor,
        retrieval_chunks=[
            FileChunk(file_path="defs/X", content="class X: ..."),
        ],
    )
    # PSI version should contain the definition, non-PSI should not
    assert "class X:" not in prompt_no_psi
    assert "class X:" in prompt_with_psi


def test_build_prompt_adaptive_context_reduces_with_chunks():
    # Create a file large enough that context window matters
    big_file = "\n".join(f"line_{i} = {i}" for i in range(400))
    cursor = big_file.index("line_200")
    chunks = [FileChunk(file_path="defs/Y", content="class Y: ...")]

    prompt_with, *_ = build_prompt("big.py", big_file, cursor, retrieval_chunks=chunks)
    prompt_without, *_ = build_prompt("big.py", big_file, cursor)

    # With PSI chunks, context is reduced so overall prompt should be shorter
    # (the PSI content is small, but the context reduction removes ~100 lines)
    assert len(prompt_with) < len(prompt_without)
