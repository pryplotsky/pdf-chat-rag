import uuid
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.database import get_document, init_db, list_documents
from app.metrics import MetricsTracker
from app.rag import RAGService
from app.schemas import ChatRequest, ChatResponse, DeleteResponse, DocumentMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="PDF Chat RAG")


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService()


@lru_cache
def get_metrics_tracker() -> MetricsTracker:
    return MetricsTracker()


@app.on_event("startup")
def startup() -> None:
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    max_size = settings.MAX_PDF_SIZE_MB * 1024 * 1024
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"PDF must be {settings.MAX_PDF_SIZE_MB} MB or smaller.",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{uuid.uuid4().hex}_{filename}"
    stored_path.write_bytes(contents)

    try:
        return rag_service.ingest_pdf(str(stored_path), filename)
    except ValueError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentMetadata])
def documents() -> list[dict]:
    return list_documents()


@app.get("/documents/{document_id}", response_model=DocumentMetadata)
def document(document_id: str) -> dict:
    metadata = get_document(document_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return metadata


@app.get("/documents/{document_id}/chunks")
def document_chunks(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> list[dict]:
    try:
        return rag_service.get_document_chunks(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/documents/{document_id}/chat", response_model=ChatResponse)
def chat(
    document_id: str,
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    try:
        return rag_service.answer_question(document_id, request.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    try:
        deleted = rag_service.delete_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if deleted is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "document_id": document_id}


@app.get("/analytics/summary")
def analytics_summary(
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker),
) -> dict:
    return metrics_tracker.get_analytics_summary()


@app.get("/analytics/chats")
def analytics_chats(
    limit: int = 20,
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker),
) -> list[dict]:
    return metrics_tracker.get_recent_chat_logs(limit=limit)


@app.get("/analytics/ingestion")
def analytics_ingestion(
    limit: int = 20,
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker),
) -> list[dict]:
    return metrics_tracker.get_document_metrics(limit=limit)
