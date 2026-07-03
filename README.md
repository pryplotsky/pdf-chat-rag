# PDF Chat RAG

Local PDF question-answering application that turns uploaded documents into a searchable knowledge base and answers business questions with cited sources, retrieval diagnostics, and analytics.

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) workflow using FastAPI, Streamlit, Qdrant, SentenceTransformers, BM25, CrossEncoder reranking, SQLite analytics, and a local Ollama LLM.

## Business Problem

Teams often store important information in PDFs: candidate profiles, interview notes, policies, reports, technical documentation, contracts, and internal knowledge documents. Finding the right answer manually is slow, especially when users need evidence from the original document.

This application solves that by letting users:

- Upload a PDF.
- Ask natural-language questions.
- Retrieve the most relevant document passages.
- Generate a grounded answer using only the retrieved context.
- Inspect the exact source chunks used for the answer.
- Track latency, token usage, and pipeline performance.

The result is a local, privacy-friendly RAG MVP that can be used to prototype document search, HR screening support, internal knowledge assistants, and PDF-based Q&A workflows.

## Demo

### Chat With a PDF

The chat view shows the selected document, user question, generated answer, timing metrics, enhanced retrieval queries, and cited source chunks.

![Chat answer demo](docs/images/chat_answer_demo.png)

### RAG Analytics Dashboard

The analytics view tracks response time, preprocessing time, LLM latency, token usage, retrieved chunks, recent questions, and document ingestion metrics.

![Analytics demo](docs/images/analytics_demo.png)

## Architecture

- PDF upload and ingestion.
- Page-level text extraction with PyMuPDF.
- Dense vector retrieval with Qdrant.
- Sparse keyword retrieval with BM25.
- Query enhancement for stronger search coverage.
- Reciprocal Rank Fusion to combine dense and sparse results.
- CrossEncoder reranking for better final chunk ordering.
- Local answer generation with Ollama.
- Page-level citations and source quotes.
- Chunk inspection for debugging retrieval quality.
- SQLite analytics dashboard for latency and token tracking.
- Fully local Docker-based setup.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Vector database | Qdrant |
| Metadata and analytics | SQLite |
| PDF parsing | PyMuPDF |
| Embeddings | SentenceTransformers MiniLM |
| Sparse retrieval | BM25 with `rank-bm25` |
| Reranking | CrossEncoder |
| Local LLM | Ollama |
| Deployment | Docker Compose |

## Local Setup on Windows

1. Install Docker Desktop.
2. Install Ollama directly on Windows.
3. Pull the local model:

```powershell
ollama pull qwen2.5:1.5b
```

4. Create your environment file:

```powershell
Copy-Item .env.example .env
```

5. Start the app:

```powershell
docker compose up -d --build
```

6. Open the Streamlit UI:

```text
http://localhost:8501
```

## Future Improvements

- OCR for scanned PDFs.
- Multi-document chat.
- Quote highlighting inside the source document.
- Authentication and role-based access.
- Postgres instead of SQLite for production metadata.
- Langfuse, Prometheus, or OpenTelemetry integration.
- GPU-backed inference for lower latency.
