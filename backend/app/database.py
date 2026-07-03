import sqlite3
from pathlib import Path

from app.config import settings
from app.metrics import MetricsTracker


def _connection() -> sqlite3.Connection:
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                num_pages INTEGER NOT NULL,
                num_chunks INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document
            ON chunks (document_id, page_number, chunk_index)
            """
        )
    MetricsTracker().init_tables()


def create_document(
    document_id: str,
    filename: str,
    upload_path: str,
    created_at: str,
    num_pages: int,
    num_chunks: int,
) -> dict:
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                id,
                filename,
                upload_path,
                created_at,
                num_pages,
                num_chunks
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                upload_path,
                created_at,
                num_pages,
                num_chunks,
            ),
        )
    document = get_document(document_id)
    if document is None:
        raise RuntimeError("Document metadata was not saved.")
    return document


def list_documents() -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, upload_path, created_at, num_pages, num_chunks
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_document(document_id: str) -> dict | None:
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT id, filename, upload_path, created_at, num_pages, num_chunks
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def create_chunks(document_id: str, chunks: list[dict]) -> None:
    with _connection() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id,
                document_id,
                chunk_id,
                page_number,
                chunk_index,
                text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"{document_id}:{chunk['chunk_id']}",
                    document_id,
                    chunk["chunk_id"],
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["text"],
                )
                for chunk in chunks
            ],
        )


def list_chunks(document_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, page_number, chunk_index, text
            FROM chunks
            WHERE document_id = ?
            ORDER BY page_number ASC, chunk_index ASC
            """,
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(document_id: str) -> dict | None:
    document = get_document(document_id)
    if document is None:
        return None

    with _connection() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return document
