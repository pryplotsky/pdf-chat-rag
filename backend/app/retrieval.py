def _result_key(result: dict) -> str:
    chunk_id = result.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"page-{result.get('page_number')}-chunk-{result.get('chunk_index')}"


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    rrf_k: int = 60,
) -> list[dict]:
    fused: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = _result_key(result)
            if key not in fused:
                fused[key] = {
                    **result,
                    "fusion_score": 0.0,
                    "retrieval_methods": [],
                    "raw_scores": {},
                }

            method = result.get("retrieval_method", "unknown")
            fused[key]["fusion_score"] += 1.0 / (rrf_k + rank)
            if method not in fused[key]["retrieval_methods"]:
                fused[key]["retrieval_methods"].append(method)

            previous_score = fused[key]["raw_scores"].get(method)
            current_score = float(result.get("score", 0.0))
            if previous_score is None or current_score > previous_score:
                fused[key]["raw_scores"][method] = current_score

    results = list(fused.values())
    results.sort(key=lambda item: item["fusion_score"], reverse=True)
    return results
