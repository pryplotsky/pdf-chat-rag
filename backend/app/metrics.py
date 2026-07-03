import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator

from app.config import settings


def _connection() -> sqlite3.Connection:
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, column_type in columns.items():
        if column_name not in existing:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )


@contextmanager
def time_block(name: str, timings: dict[str, float] | None = None) -> Iterator[dict[str, float]]:
    result = {"name": name, "elapsed_ms": 0.0}
    start = perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (perf_counter() - start) * 1000
        if timings is not None:
            timings[name] = result["elapsed_ms"]


class MetricsTracker:
    def __init__(self) -> None:
        self.init_tables()

    def init_tables(self) -> None:
        with _connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_metrics (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    filename TEXT,
                    num_pages INTEGER,
                    num_chunks INTEGER,
                    total_preprocessing_time_ms REAL,
                    pdf_parsing_time_ms REAL,
                    chunking_time_ms REAL,
                    embedding_time_ms REAL,
                    qdrant_upsert_time_ms REAL,
                    created_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    question TEXT,
                    model_name TEXT,
                    total_response_time_ms REAL,
                    query_enhancement_time_ms REAL DEFAULT 0,
                    query_embedding_time_ms REAL,
                    qdrant_search_time_ms REAL,
                    bm25_search_time_ms REAL DEFAULT 0,
                    fusion_time_ms REAL DEFAULT 0,
                    rerank_time_ms REAL DEFAULT 0,
                    prompt_build_time_ms REAL,
                    llm_generation_time_ms REAL,
                    prompt_tokens INTEGER,
                    generated_tokens INTEGER,
                    total_tokens INTEGER,
                    retrieved_chunks INTEGER,
                    avg_relevance_score REAL,
                    created_at TEXT
                )
                """
            )
            _ensure_columns(
                connection,
                "chat_metrics",
                {
                    "query_enhancement_time_ms": "REAL DEFAULT 0",
                    "bm25_search_time_ms": "REAL DEFAULT 0",
                    "fusion_time_ms": "REAL DEFAULT 0",
                    "rerank_time_ms": "REAL DEFAULT 0",
                },
            )

    def log_ingestion_metrics(
        self,
        document_id: str,
        filename: str,
        num_pages: int,
        num_chunks: int,
        total_preprocessing_time_ms: float,
        pdf_parsing_time_ms: float,
        chunking_time_ms: float,
        embedding_time_ms: float,
        qdrant_upsert_time_ms: float,
    ) -> None:
        with _connection() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_metrics (
                    id,
                    document_id,
                    filename,
                    num_pages,
                    num_chunks,
                    total_preprocessing_time_ms,
                    pdf_parsing_time_ms,
                    chunking_time_ms,
                    embedding_time_ms,
                    qdrant_upsert_time_ms,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    filename,
                    num_pages,
                    num_chunks,
                    total_preprocessing_time_ms,
                    pdf_parsing_time_ms,
                    chunking_time_ms,
                    embedding_time_ms,
                    qdrant_upsert_time_ms,
                    _utc_now(),
                ),
            )

    def log_chat_metrics(
        self,
        request_id: str,
        document_id: str,
        question: str,
        model_name: str,
        total_response_time_ms: float,
        query_enhancement_time_ms: float,
        query_embedding_time_ms: float,
        qdrant_search_time_ms: float,
        bm25_search_time_ms: float,
        fusion_time_ms: float,
        rerank_time_ms: float,
        prompt_build_time_ms: float,
        llm_generation_time_ms: float,
        prompt_tokens: int,
        generated_tokens: int,
        total_tokens: int,
        retrieved_chunks: int,
        avg_relevance_score: float,
    ) -> None:
        with _connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_metrics (
                    id,
                    document_id,
                    question,
                    model_name,
                    total_response_time_ms,
                    query_enhancement_time_ms,
                    query_embedding_time_ms,
                    qdrant_search_time_ms,
                    bm25_search_time_ms,
                    fusion_time_ms,
                    rerank_time_ms,
                    prompt_build_time_ms,
                    llm_generation_time_ms,
                    prompt_tokens,
                    generated_tokens,
                    total_tokens,
                    retrieved_chunks,
                    avg_relevance_score,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    document_id,
                    question,
                    model_name,
                    total_response_time_ms,
                    query_enhancement_time_ms,
                    query_embedding_time_ms,
                    qdrant_search_time_ms,
                    bm25_search_time_ms,
                    fusion_time_ms,
                    rerank_time_ms,
                    prompt_build_time_ms,
                    llm_generation_time_ms,
                    prompt_tokens,
                    generated_tokens,
                    total_tokens,
                    retrieved_chunks,
                    avg_relevance_score,
                    _utc_now(),
                ),
            )

    def get_analytics_summary(self) -> dict:
        with _connection() as connection:
            documents = connection.execute("SELECT COUNT(*) AS value FROM documents").fetchone()
            chats = connection.execute("SELECT COUNT(*) AS value FROM chat_metrics").fetchone()
            ingestion = connection.execute(
                """
                SELECT AVG(total_preprocessing_time_ms) AS avg_preprocessing_time_ms
                FROM ingestion_metrics
                """
            ).fetchone()
            chat = connection.execute(
                """
                SELECT
                    AVG(total_response_time_ms) AS avg_response_time_ms,
                    AVG(query_enhancement_time_ms) AS avg_query_enhancement_time_ms,
                    AVG(query_embedding_time_ms) AS avg_query_embedding_time_ms,
                    AVG(qdrant_search_time_ms) AS avg_qdrant_search_time_ms,
                    AVG(bm25_search_time_ms) AS avg_bm25_search_time_ms,
                    AVG(fusion_time_ms) AS avg_fusion_time_ms,
                    AVG(rerank_time_ms) AS avg_rerank_time_ms,
                    AVG(llm_generation_time_ms) AS avg_llm_generation_time_ms,
                    AVG(prompt_tokens) AS avg_prompt_tokens,
                    AVG(generated_tokens) AS avg_generated_tokens,
                    AVG(total_tokens) AS avg_total_tokens,
                    AVG(retrieved_chunks) AS avg_retrieved_chunks
                FROM chat_metrics
                """
            ).fetchone()

        summary = {
            "total_documents": int(documents["value"] or 0),
            "total_questions": int(chats["value"] or 0),
            "avg_preprocessing_time_ms": 0.0,
            "avg_response_time_ms": 0.0,
            "avg_query_enhancement_time_ms": 0.0,
            "avg_query_embedding_time_ms": 0.0,
            "avg_qdrant_search_time_ms": 0.0,
            "avg_bm25_search_time_ms": 0.0,
            "avg_fusion_time_ms": 0.0,
            "avg_rerank_time_ms": 0.0,
            "avg_llm_generation_time_ms": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_generated_tokens": 0.0,
            "avg_total_tokens": 0.0,
            "avg_retrieved_chunks": 0.0,
        }
        if ingestion:
            summary["avg_preprocessing_time_ms"] = float(
                ingestion["avg_preprocessing_time_ms"] or 0.0
            )
        if chat:
            for key in (
                "avg_response_time_ms",
                "avg_query_enhancement_time_ms",
                "avg_query_embedding_time_ms",
                "avg_qdrant_search_time_ms",
                "avg_bm25_search_time_ms",
                "avg_fusion_time_ms",
                "avg_rerank_time_ms",
                "avg_llm_generation_time_ms",
                "avg_prompt_tokens",
                "avg_generated_tokens",
                "avg_total_tokens",
                "avg_retrieved_chunks",
            ):
                summary[key] = float(chat[key] or 0.0)
        return summary

    def get_recent_chat_logs(self, limit: int = 20) -> list[dict]:
        with _connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    document_id,
                    question,
                    model_name,
                    total_response_time_ms,
                    query_enhancement_time_ms,
                    query_embedding_time_ms,
                    qdrant_search_time_ms,
                    bm25_search_time_ms,
                    fusion_time_ms,
                    rerank_time_ms,
                    prompt_build_time_ms,
                    llm_generation_time_ms,
                    prompt_tokens,
                    generated_tokens,
                    total_tokens,
                    retrieved_chunks,
                    avg_relevance_score,
                    created_at
                FROM chat_metrics
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document_metrics(self, limit: int = 20) -> list[dict]:
        with _connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    document_id,
                    filename,
                    num_pages,
                    num_chunks,
                    total_preprocessing_time_ms,
                    pdf_parsing_time_ms,
                    chunking_time_ms,
                    embedding_time_ms,
                    qdrant_upsert_time_ms,
                    created_at
                FROM ingestion_metrics
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
