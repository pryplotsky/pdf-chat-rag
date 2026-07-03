# PDF Chat RAG

A local open-source RAG application for chatting with PDF documents using FastAPI, Streamlit, Qdrant, SentenceTransformers, and Ollama.

## Features

- PDF upload
- Page-level text extraction
- Hybrid dense and sparse retrieval
- Query enhancement with rewritten and step-back queries
- Reciprocal Rank Fusion with chunk deduplication
- CrossEncoder reranking
- Chunk inspection in the Documents tab
- SQLite RAG analytics dashboard
- Local LLM answers
- Page-level citations
- Source quotes
- Hallucination-safe fallback

## Analytics

The backend logs ingestion and chat metrics to SQLite and exposes them through API endpoints.
The Streamlit app includes an `Analytics` tab with summary cards and recent logs.

Tracked ingestion metrics:

- Total preprocessing time
- PDF parsing time
- Chunking time
- Embedding time
- Qdrant upsert time
- Pages and chunks per document

Tracked chat metrics:

- Total response time
- Query embedding time
- Qdrant search time
- Prompt build time
- LLM generation time
- Prompt tokens
- Generated tokens
- Total tokens
- Retrieved chunks
- Average relevance score

## Architecture

```text
Browser
  |
  v
Streamlit frontend :8501
  |
  v
FastAPI backend :8000
  |       |         |
  |       |         +--> Ollama on Windows :11434
  |       |
  |       +--> SQLite metadata + chunk text /app/data/app.db
  |
  +--> Qdrant vector DB :6333

PDF upload -> PyMuPDF pages -> character chunks -> MiniLM embeddings -> Qdrant
           -> chunk text + metadata -> SQLite

Question -> query enhancement
         -> dense Qdrant search for each query variant
         -> sparse BM25 search for each query variant
         -> Reciprocal Rank Fusion + dedupe
         -> CrossEncoder reranking
         -> top chunks -> Ollama prompt -> cited answer
```

## Retrieval Pipeline

The app uses an enhanced local RAG pipeline optimized for a CPU-only laptop:

1. The original question is enhanced into up to 3 query variants.
2. Each query variant is embedded with MiniLM.
3. Qdrant retrieves the top 25 dense vector candidates per query variant.
4. BM25 retrieves the top 25 sparse keyword candidates per query variant from the SQLite chunk table.
5. Dense and sparse ranked lists are merged with Reciprocal Rank Fusion.
6. Duplicate chunks are removed during fusion.
7. The top 25 fused candidates are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
8. The final top 3 chunks are sent to Ollama as the only answer context.

## Local Setup on Windows

1. Install Docker Desktop.
2. Install Ollama directly on Windows.
3. Pull the small local model:

```powershell
ollama pull qwen2.5:1.5b
```

4. Create your environment file:

```powershell
Copy-Item .env.example .env
```

5. Start Qdrant, the FastAPI backend, and the Streamlit frontend:

```powershell
docker compose up -d --build
```

6. Open the app:

```text
http://localhost:8501
```

Backend docs:

```text
http://localhost:8000/docs
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

Ollama is intentionally not included in Docker Compose. The backend calls Ollama through `http://host.docker.internal:11434`, which lets the containers reach the Windows host.

## API Endpoints

- `GET /health` returns backend health.
- `POST /documents/upload` uploads and ingests a PDF.
- `GET /documents` lists uploaded documents.
- `GET /documents/{document_id}` returns document metadata.
- `GET /documents/{document_id}/chunks` returns stored chunks for one document.
- `POST /documents/{document_id}/chat` answers a question about one document.
- `DELETE /documents/{document_id}` deletes document metadata, vectors, and the uploaded file.
- `GET /analytics/summary` returns aggregate analytics.
- `GET /analytics/chats` returns recent chat metrics.
- `GET /analytics/ingestion` returns recent ingestion metrics.

## Retrieval Settings

- `DENSE_TOP_K=25` controls Qdrant vector candidates.
- `SPARSE_TOP_K=25` controls BM25 candidates.
- `RRF_K=60` controls Reciprocal Rank Fusion smoothing.
- `RERANK_TOP_N=25` controls how many fused candidates go to the CrossEncoder.
- `FINAL_TOP_K=3` controls how many chunks are sent to Ollama.
- `QUERY_ENHANCEMENT_ENABLED=true` enables enhanced query variants.
- `QUERY_VARIANTS=3` controls the maximum number of query variants.
- `RERANKING_ENABLED=true` enables CrossEncoder reranking.

## Limits

- PDF size limit: 20 MB.
- Page limit: 80 pages.
- Final answer context: top 3 reranked chunks.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Sparse retriever: BM25 with `rank-bm25`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- PyTorch is pinned to a CPU-only wheel for smaller Docker builds on laptops without a dedicated GPU.
- Recommended Ollama model: `qwen2.5:1.5b`.

## Limitations

- No OCR yet.
- No table extraction.
- Uses a small local model.
- Query enhancement adds one extra local Ollama call per question.
- CrossEncoder reranking is CPU-friendly but still slower than dense-only retrieval.
- Limited to 80 pages.
- Not for confidential documents.

## Future Improvements

- OCR
- Multi-document chat
- Quote highlighting
- Evaluation dataset
- Langfuse or OpenTelemetry integration
- Cloud deployment
