from hermen.chunking import chunk_text


def test_chunk_text_splits_with_overlap() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunks = chunk_text(text, chunk_size=24, chunk_overlap=6)

    assert len(chunks) >= 2
    assert chunks[0].text.startswith("alpha beta")
    assert "epsilon" in chunks[1].text
