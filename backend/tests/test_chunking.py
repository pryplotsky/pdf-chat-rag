from app.chunking import chunk_pages


def test_chunking_creates_non_empty_chunks() -> None:
    pages = [{"page_number": 1, "text": "alpha beta gamma delta epsilon zeta"}]

    chunks = chunk_pages(pages, chunk_size=16, chunk_overlap=5)

    assert chunks
    assert all(chunk["text"] for chunk in chunks)


def test_chunking_preserves_page_number() -> None:
    pages = [{"page_number": 4, "text": "one two three four five six seven"}]

    chunks = chunk_pages(pages, chunk_size=12, chunk_overlap=3)

    assert {chunk["page_number"] for chunk in chunks} == {4}


def test_chunking_overlap_works() -> None:
    pages = [{"page_number": 1, "text": "abcdefghijklmnopqrstuvwxyz"}]

    chunks = chunk_pages(pages, chunk_size=10, chunk_overlap=3)

    assert chunks[0]["text"] == "abcdefghij"
    assert chunks[1]["text"].startswith("hij")
