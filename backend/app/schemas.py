from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    upload_path: str
    created_at: str
    num_pages: int
    num_chunks: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceQuote(BaseModel):
    page_number: int
    quote: str
    score: float


class ChatMetrics(BaseModel):
    total_response_time_ms: float
    query_enhancement_time_ms: float = 0.0
    query_embedding_time_ms: float
    qdrant_search_time_ms: float
    bm25_search_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    prompt_build_time_ms: float
    llm_generation_time_ms: float
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    retrieved_chunks: int
    avg_relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceQuote]
    enhanced_queries: list[str] = Field(default_factory=list)
    metrics: ChatMetrics | None = None


class DeleteResponse(BaseModel):
    status: str
    document_id: str
