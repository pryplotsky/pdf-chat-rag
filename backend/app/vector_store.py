import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings


class QdrantVectorStore:
    def __init__(self, vector_size: int) -> None:
        self.collection_name = settings.QDRANT_COLLECTION
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        exists = self.client.collection_exists(collection_name=self.collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk['chunk_id']}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "chunk_id": chunk["chunk_id"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        document_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            results = response.points
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

        formatted = []
        for result in results:
            payload = result.payload or {}
            formatted.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text", ""),
                    "page_number": payload.get("page_number"),
                    "chunk_index": payload.get("chunk_index"),
                    "score": float(result.score),
                    "retrieval_method": "dense",
                }
            )
        return formatted

    def list_document_chunks(self, document_id: str) -> list[dict]:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        chunks: list[dict] = []
        for record in records:
            payload = record.payload or {}
            page_number = int(payload.get("page_number") or 0)
            chunk_index = int(payload.get("chunk_index") or 0)
            chunk_id = payload.get("chunk_id") or f"page-{page_number}-chunk-{chunk_index}"
            text = str(payload.get("text") or "")
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": str(chunk_id),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )

        chunks.sort(key=lambda item: (item["page_number"], item["chunk_index"]))
        return chunks

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
