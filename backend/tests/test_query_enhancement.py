from app.query_enhancement import build_query_enhancement_prompt, dedupe_queries, is_english_query


def test_query_enhancement_prompt_requests_rewrite_and_step_back() -> None:
    prompt = build_query_enhancement_prompt("What are the payment terms?")

    assert "rewritten_query" in prompt
    assert "step_back_query" in prompt
    assert "decomposed_queries" in prompt
    assert "English only" in prompt
    assert "Enhance the user's question" in prompt
    assert "valid JSON only" in prompt


def test_dedupe_queries_preserves_order() -> None:
    queries = dedupe_queries(["Payment terms", " payment   terms ", "Late fees"])

    assert queries == ["Payment terms", "Late fees"]


def test_non_english_cjk_queries_are_filtered() -> None:
    queries = dedupe_queries(
        ["What tools do I know?", "我有哪些工具？", "мої інструменти?"]
    )

    assert queries == ["What tools do I know?"]
    assert is_english_query("What tools do I know?")
    assert not is_english_query("我有哪些工具？")
    assert not is_english_query("мої інструменти?")
