import json
import re

from app.config import settings


NON_ENGLISH_SCRIPT_PATTERN = re.compile(
    r"[\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f"
    r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
)


def build_query_enhancement_prompt(question: str) -> str:
    return f"""Task:
Enhance the user's question into better retrieval queries for a PDF RAG system.

Language:
- Write every generated query in English only.
- Use English retrieval wording even when the user question contains another language.
- Do not output Chinese, Japanese, Korean, Cyrillic, Arabic, Hebrew, or other non-English scripts.

Rules:
- Preserve the user's original intent.
- Do not answer the question.
- Do not add facts, names, tools, dates, or assumptions not present in the question.
- Generate document-content search phrases only.
- Do not generate meta queries about retrieval, searching, how to find information, or the user's identity.
- Prefer concise search-friendly phrases.
- Return valid JSON only. No markdown. No commentary.

JSON schema:
{{
  "rewritten_query": "A clearer English version of the user question.",
  "step_back_query": "A broader English query for the underlying topic.",
  "decomposed_queries": [
    "Optional smaller English sub-question 1",
    "Optional smaller English sub-question 2"
  ]
}}

User question:
{question}

JSON:"""


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def is_english_query(query: str) -> bool:
    return not NON_ENGLISH_SCRIPT_PATTERN.search(query)


def is_content_query(query: str) -> bool:
    normalized = " ".join(query.casefold().split())
    blocked_phrases = (
        "how can i retrieve",
        "how do i retrieve",
        "retrieve information",
        "how can i find",
        "how do i find",
        "search for",
        "user's identity",
        "user identity",
    )
    return not any(phrase in normalized for phrase in blocked_phrases)


def dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        cleaned = " ".join(str(query).split())
        key = cleaned.casefold()
        if cleaned and is_english_query(cleaned) and is_content_query(cleaned) and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


class QueryEnhancer:
    def __init__(self, llm) -> None:
        self.llm = llm

    def enhance(self, question: str) -> list[str]:
        original = " ".join(question.split())
        if not original:
            return []
        if not settings.QUERY_ENHANCEMENT_ENABLED:
            return [original]

        prompt = build_query_enhancement_prompt(original)
        try:
            raw_response = self.llm.generate_text(prompt, timeout=45)
            parsed = _extract_json(raw_response)
        except Exception:
            return [original]

        candidates = [
            original,
            parsed.get("rewritten_query", ""),
            parsed.get("step_back_query", ""),
        ]
        decomposed = parsed.get("decomposed_queries", [])
        if isinstance(decomposed, list):
            candidates.extend(decomposed)

        return dedupe_queries(candidates)[: settings.QUERY_VARIANTS]
