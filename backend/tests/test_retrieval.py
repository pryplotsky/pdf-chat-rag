from app.retrieval import reciprocal_rank_fusion


def test_rrf_deduplicates_chunks() -> None:
    dense = [
        {"chunk_id": "a", "text": "alpha", "score": 0.9, "retrieval_method": "dense"},
        {"chunk_id": "b", "text": "beta", "score": 0.8, "retrieval_method": "dense"},
    ]
    sparse = [
        {"chunk_id": "a", "text": "alpha", "score": 3.0, "retrieval_method": "sparse"},
    ]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "a"
    assert sorted(fused[0]["retrieval_methods"]) == ["dense", "sparse"]


def test_rrf_orders_by_fused_rank() -> None:
    dense = [
        {"chunk_id": "a", "text": "alpha", "score": 0.9, "retrieval_method": "dense"},
        {"chunk_id": "b", "text": "beta", "score": 0.8, "retrieval_method": "dense"},
    ]
    sparse = [
        {"chunk_id": "b", "text": "beta", "score": 4.0, "retrieval_method": "sparse"},
    ]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0]["chunk_id"] == "b"
