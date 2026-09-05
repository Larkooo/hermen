from hermen.chunking import chunk_text
import subprocess
import sys
import os
from pathlib import Path


def test_chunk_text_splits_with_overlap() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunks = chunk_text(text, chunk_size=24, chunk_overlap=6)

    assert len(chunks) >= 2
    assert chunks[0].text.startswith("alpha beta")
    assert "epsilon" in chunks[1].text


def test_overlap_after_short_word_makes_progress() -> None:
    # Bound the regression check: this input previously hung indefinitely.
    code = (
        "from hermen.chunking import chunk_text; "
        "parts = chunk_text('a ' + 'b' * 40, chunk_size=20, chunk_overlap=15); "
        "assert parts[0].text == 'a'; assert parts[-1].text.endswith('b'); "
        "assert len(parts) < 20"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=5,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
