import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app import database
from app.chunking import chunk_pages
from app.config import settings
from app.embeddings import EmbeddingService
from app.llm import FALLBACK_ANSWER, LocalLLM
from app.metrics import MetricsTracker, time_block
from app.pdf_parser import extract_pdf_pages
from app.query_enhancement import QueryEnhancer, is_english_query
from app.reranker import CrossEncoderReranker
from app.retrieval import reciprocal_rank_fusion
from app.sparse_retriever import BM25Retriever
from app.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)
ENGLISH_ONLY_MESSAGE = "Please ask questions in English only."


def _chat_metrics_payload(
    total_response_time_ms: float,
    timings: dict[str, float] | None = None,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    total_tokens: int = 0,
    retrieved_chunks: int = 0,
    avg_relevance_score: float = 0.0,
) -> dict:
    timings = timings or {}
    return {
        "total_response_time_ms": total_response_time_ms,
        "query_enhancement_time_ms": timings.get("query_enhancement_time_ms", 0.0),
        "query_embedding_time_ms": timings.get("query_embedding_time_ms", 0.0),
        "qdrant_search_time_ms": timings.get("qdrant_search_time_ms", 0.0),
        "bm25_search_time_ms": timings.get("bm25_search_time_ms", 0.0),
        "fusion_time_ms": timings.get("fusion_time_ms", 0.0),
        "rerank_time_ms": timings.get("rerank_time_ms", 0.0),
        "prompt_build_time_ms": timings.get("prompt_build_time_ms", 0.0),
        "llm_generation_time_ms": timings.get("llm_generation_time_ms", 0.0),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "retrieved_chunks": retrieved_chunks,
        "avg_relevance_score": avg_relevance_score,
    }


def _format_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "page_number": chunk["page_number"],
            "quote": chunk["text"][:500],
            "score": float(
                chunk.get(
                    "relevance_score",
                    chunk.get("fusion_score", chunk.get("score", 0.0)),
                )
            ),
        }
        for chunk in chunks
    ]


class RAGService:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store = QdrantVectorStore(vector_size=self.embedding_service.dimension)
        self.llm = LocalLLM()
        self.query_enhancer = QueryEnhancer(self.llm)
        self.sparse_retriever = BM25Retriever()
        self.reranker = CrossEncoderReranker()
        self.metrics_tracker = MetricsTracker()

    def get_document_chunks(self, document_id: str) -> list[dict]:
        document = database.get_document(document_id)
        if document is None:
            raise ValueError("Document not found.")

        chunks = database.list_chunks(document_id)
        if chunks:
            return chunks

        chunks = self.vector_store.list_document_chunks(document_id)
        if chunks:
            database.create_chunks(document_id, chunks)
            logger.info(
                "chunks.backfill_from_qdrant document_id=%s chunks=%d",
                document_id,
                len(chunks),
            )
        return chunks

    def ingest_pdf(self, file_path: str, filename: str) -> dict:
        total_start = perf_counter()
        timings: dict[str, float] = {}
        document_id = str(uuid.uuid4())
        logger.info("ingest.start document_id=%s filename=%s", document_id, filename)

        with time_block("pdf_parsing_time_ms", timings):
            pages = extract_pdf_pages(file_path, settings.MAX_PAGES)
        logger.info(
            "ingest.extract_pdf done pages=%d elapsed=%.1fms",
            len(pages),
            timings["pdf_parsing_time_ms"],
        )
        if not pages:
            raise ValueError("No readable text was found in the PDF.")

        with time_block("chunking_time_ms", timings):
            chunks = chunk_pages(
                pages,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
        logger.info(
            "ingest.chunk_pages done chunks=%d elapsed=%.1fms",
            len(chunks),
            timings["chunking_time_ms"],
        )
        if not chunks:
            raise ValueError("No text chunks could be created from the PDF.")

        with time_block("embedding_time_ms", timings):
            embeddings = self.embedding_service.embed_texts([chunk["text"] for chunk in chunks])
        logger.info(
            "ingest.embed_chunks done embeddings=%d elapsed=%.1fms",
            len(embeddings),
            timings["embedding_time_ms"],
        )

        with time_block("qdrant_upsert_time_ms", timings):
            self.vector_store.upsert_chunks(document_id, chunks, embeddings)
        logger.info(
            "ingest.qdrant_upsert done elapsed=%.1fms",
            timings["qdrant_upsert_time_ms"],
        )

        created_at = datetime.now(timezone.utc).isoformat()
        step_start = perf_counter()
        document = database.create_document(
            document_id=document_id,
            filename=filename,
            upload_path=str(Path(file_path)),
            created_at=created_at,
            num_pages=len(pages),
            num_chunks=len(chunks),
        )
        database.create_chunks(document_id, chunks)
        logger.info("ingest.sqlite_save done elapsed=%.3fs", perf_counter() - step_start)
        timings["total_preprocessing_time_ms"] = (perf_counter() - total_start) * 1000
        self.metrics_tracker.log_ingestion_metrics(
            document_id=document_id,
            filename=filename,
            num_pages=len(pages),
            num_chunks=len(chunks),
            total_preprocessing_time_ms=timings["total_preprocessing_time_ms"],
            pdf_parsing_time_ms=timings["pdf_parsing_time_ms"],
            chunking_time_ms=timings["chunking_time_ms"],
            embedding_time_ms=timings["embedding_time_ms"],
            qdrant_upsert_time_ms=timings["qdrant_upsert_time_ms"],
        )
        logger.info(
            "ingest.done document_id=%s total_elapsed=%.1fms",
            document_id,
            timings["total_preprocessing_time_ms"],
        )
        return {
            "document_id": document["id"],
            "filename": document["filename"],
            "num_pages": document["num_pages"],
            "num_chunks": document["num_chunks"],
            "timing": timings,
        }

    def answer_question(self, document_id: str, question: str) -> dict:
        total_start = perf_counter()
        timings: dict[str, float] = {}
        request_id = str(uuid.uuid4())
        logger.info("chat.start document_id=%s question=%r", document_id, question)
        document = database.get_document(document_id)
        if document is None:
            raise ValueError("Document not found.")
        if not is_english_query(question):
            total_response_time_ms = (perf_counter() - total_start) * 1000
            logger.info("chat.rejected_non_english total_elapsed=%.1fms", total_response_time_ms)
            return {
                "answer": ENGLISH_ONLY_MESSAGE,
                "sources": [],
                "enhanced_queries": [],
                "metrics": _chat_metrics_payload(total_response_time_ms),
            }
        self.get_document_chunks(document_id)

        with time_block("query_enhancement_time_ms", timings):
            query_variants = self.query_enhancer.enhance(question)
        if not query_variants:
            query_variants = [question]
        logger.info(
            "chat.query_enhancement done variants=%d elapsed=%.1fms queries=%s",
            len(query_variants),
            timings["query_enhancement_time_ms"],
            query_variants,
        )

        with time_block("query_embedding_time_ms", timings):
            query_embeddings = self.embedding_service.embed_texts(query_variants)
        logger.info("chat.embed_query done elapsed=%.1fms", timings["query_embedding_time_ms"])

        dense_result_lists: list[list[dict]] = []
        with time_block("qdrant_search_time_ms", timings):
            for query_embedding in query_embeddings:
                dense_result_lists.append(
                    self.vector_store.search(
                        document_id=document_id,
                        query_embedding=query_embedding,
                        top_k=settings.DENSE_TOP_K,
                    )
                )
        dense_results_count = sum(len(results) for results in dense_result_lists)
        logger.info(
            "chat.dense_search done requested=%d returned=%d elapsed=%.1fms",
            settings.DENSE_TOP_K * len(query_variants),
            dense_results_count,
            timings["qdrant_search_time_ms"],
        )

        sparse_result_lists: list[list[dict]] = []
        with time_block("bm25_search_time_ms", timings):
            for query_variant in query_variants:
                sparse_result_lists.append(
                    self.sparse_retriever.search(
                        document_id=document_id,
                        query=query_variant,
                        top_k=settings.SPARSE_TOP_K,
                    )
                )
        sparse_results_count = sum(len(results) for results in sparse_result_lists)
        logger.info(
            "chat.bm25_search done requested=%d returned=%d elapsed=%.1fms",
            settings.SPARSE_TOP_K * len(query_variants),
            sparse_results_count,
            timings["bm25_search_time_ms"],
        )

        with time_block("fusion_time_ms", timings):
            result_lists = [*dense_result_lists, *sparse_result_lists]
            fused_chunks = reciprocal_rank_fusion(result_lists, rrf_k=settings.RRF_K)
        logger.info(
            "chat.fusion_dedupe done candidates=%d elapsed=%.1fms",
            len(fused_chunks),
            timings["fusion_time_ms"],
        )

        candidate_chunks = fused_chunks[: settings.RERANK_TOP_N]
        logger.info("chat.rerank start candidates=%d", len(candidate_chunks))
        with time_block("rerank_time_ms", timings):
            reranked_chunks = self.reranker.rerank(
                question=question,
                chunks=candidate_chunks,
                top_n=settings.RERANK_TOP_N,
            )
        logger.info(
            "chat.rerank done returned=%d elapsed=%.1fms",
            len(reranked_chunks),
            timings["rerank_time_ms"],
        )

        final_chunks = reranked_chunks[: settings.FINAL_TOP_K]
        if not final_chunks:
            total_response_time_ms = (perf_counter() - total_start) * 1000
            logger.info("chat.no_chunks total_elapsed=%.1fms", total_response_time_ms)
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "enhanced_queries": query_variants,
                "metrics": _chat_metrics_payload(total_response_time_ms, timings),
            }

        avg_relevance_score = sum(
            float(chunk.get("relevance_score", 0.0)) for chunk in final_chunks
        ) / len(final_chunks)

        with time_block("prompt_build_time_ms", timings):
            # The final prompt is built inside LocalLLM; this mirrors that cost for analytics.
            from app.llm import build_rag_prompt

            build_rag_prompt(question, final_chunks)

        with time_block("llm_generation_time_ms", timings):
            llm_result = self.llm.generate_answer(question, final_chunks)
        answer = llm_result["answer"]
        usage = llm_result["usage"]
        logger.info("chat.ollama_answer done elapsed=%.1fms", timings["llm_generation_time_ms"])
        total_response_time_ms = (perf_counter() - total_start) * 1000
        chat_metrics = _chat_metrics_payload(
            total_response_time_ms=total_response_time_ms,
            timings=timings,
            prompt_tokens=int(usage["prompt_tokens"]),
            generated_tokens=int(usage["generated_tokens"]),
            total_tokens=int(usage["total_tokens"]),
            retrieved_chunks=len(final_chunks),
            avg_relevance_score=avg_relevance_score,
        )
        self.metrics_tracker.log_chat_metrics(
            request_id=request_id,
            document_id=document_id,
            question=question,
            model_name=self.llm.model,
            **chat_metrics,
        )
        logger.info(
            "chat.done document_id=%s sources=%d total_elapsed=%.1fms",
            document_id,
            len(final_chunks),
            total_response_time_ms,
        )
        return {
            "answer": answer,
            "sources": _format_sources(final_chunks),
            "enhanced_queries": query_variants,
            "metrics": chat_metrics,
        }

    def delete_document(self, document_id: str) -> dict | None:
        document = database.get_document(document_id)
        if document is None:
            return None

        self.vector_store.delete_document(document_id)
        deleted = database.delete_document(document_id)

        upload_path = Path(document["upload_path"])
        if upload_path.exists():
            upload_path.unlink()

        return deleted
