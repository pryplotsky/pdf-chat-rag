from threading import Lock

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    _model: SentenceTransformer | None = None
    _lock = Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        if EmbeddingService._model is None:
            with EmbeddingService._lock:
                if EmbeddingService._model is None:
                    EmbeddingService._model = SentenceTransformer(self.model_name)
        return EmbeddingService._model

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine embedding dimension.")
        return int(dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
