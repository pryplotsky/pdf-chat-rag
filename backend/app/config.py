from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "pdf_chunks"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"

    SQLITE_DB_PATH: str = "/app/data/app.db"
    UPLOAD_DIR: str = "/app/data/uploads"

    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 3

    DENSE_TOP_K: int = 25
    SPARSE_TOP_K: int = 25
    RRF_K: int = 60
    RERANK_TOP_N: int = 25
    FINAL_TOP_K: int = 3

    QUERY_ENHANCEMENT_ENABLED: bool = True
    QUERY_VARIANTS: int = 3

    RERANKING_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    MAX_PDF_SIZE_MB: int = 20
    MAX_PAGES: int = 80

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
