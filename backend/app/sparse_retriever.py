import re

from rank_bm25 import BM25Okapi

from app import database


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def search(self, document_id: str, query: str, top_k: int) -> list[dict]:
        chunks = database.list_chunks(document_id)
        if not chunks or top_k <= 0:
            return []

        tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(tokenized_query)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[dict] = []
        for index, score in ranked[:top_k]:
            if score <= 0:
                continue
            chunk = chunks[index]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "score": float(score),
                    "retrieval_method": "sparse",
                }
            )
        return results
