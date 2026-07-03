from app.llm import FALLBACK_ANSWER, build_rag_prompt


def test_prompt_limits_answer_to_context() -> None:
    prompt = build_rag_prompt(
        "What is the revenue?",
        [{"page_number": 3, "text": "Revenue was 12 million."}],
    )

    assert "Use only the context" in prompt


def test_prompt_contains_fallback_message() -> None:
    prompt = build_rag_prompt("What is missing?", [])

    assert FALLBACK_ANSWER in prompt


def test_prompt_requires_page_citations() -> None:
    prompt = build_rag_prompt(
        "Where is this stated?",
        [{"page_number": 7, "text": "The policy starts in June."}],
    )

    assert "cite the page number" in prompt
    assert "[page 3]" in prompt


def test_prompt_explains_first_person_cv_questions() -> None:
    prompt = build_rag_prompt(
        "What programming languages do I know?",
        [{"page_number": 1, "text": "Programming Languages: Python, R, SQL"}],
    )

    assert "I, me, my, and mine refer to the document owner" in prompt
    assert "Programming languages are often listed under Skills" in prompt
