def _find_word_boundary(text: str, start: int, end: int, min_end: int) -> int:
    if end >= len(text) or text[end - 1].isspace():
        return end

    break_at = text.rfind(" ", min_end, end)
    if break_at == -1:
        break_at = text.rfind("\n", min_end, end)

    return break_at if break_at != -1 else end


def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    step = chunk_size - chunk_overlap
    chunks: list[dict] = []

    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("text", "")).strip()
        if not text:
            continue

        start = 0
        chunk_index = 0
        while start < len(text):
            raw_end = min(start + chunk_size, len(text))
            end = _find_word_boundary(text, start, raw_end, start + step)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"page-{page_number}-chunk-{chunk_index}",
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

            start += step

    return chunks
