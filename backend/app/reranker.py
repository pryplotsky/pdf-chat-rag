from threading import Lock
from math import exp

from app.config import settings


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1 / (1 + z)
    z = exp(value)
    return z / (1 + z)


class CrossEncoderReranker:
    _model = None
    _lock = Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL

    def _load_model(self):
        if CrossEncoderReranker._model is None:
            with CrossEncoderReranker._lock:
                if CrossEncoderReranker._model is None:
                    from sentence_transformers import CrossEncoder

                    CrossEncoderReranker._model = CrossEncoder(self.model_name)
        return CrossEncoderReranker._model

    def rerank(self, question: str, chunks: list[dict], top_n: int) -> list[dict]:
        if not settings.RERANKING_ENABLED or not chunks or top_n <= 0:
            return chunks[:top_n]

        try:
            model = self._load_model()
            scores = model.predict([(question, chunk["text"]) for chunk in chunks])
        except Exception:
            return chunks[:top_n]

        reranked: list[dict] = []
        for chunk, score in zip(chunks, scores, strict=True):
            raw_score = float(score)
            enriched = {
                **chunk,
                "rerank_score": raw_score,
                "relevance_score": sigmoid(raw_score),
                "score": raw_score,
            }
            reranked.append(enriched)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_n]
