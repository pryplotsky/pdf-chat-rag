FALLBACK_ANSWER = "I could not find this information in the document."


def _ns_to_ms(value: int | float | None) -> float:
    return float(value or 0) / 1_000_000


def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"Source {index} [page {chunk['page_number']}]\n{chunk['text']}"
        for index, chunk in enumerate(context_chunks, start=1)
    )
    return f"""You are a document question-answering assistant.

Use only the context below to answer the question.
Do not use outside knowledge.

The user is asking about the person, organization, or subject described in the document.
In first-person questions, words like I, me, my, and mine refer to the document owner or main subject.

Extract direct facts from the context when they are present.
For CVs and resumes:
- A name is often in the contact/header area.
- Programming languages are often listed under Skills or Programming Languages.
- Tools may be listed under Tools, Tech stack, Skills, or DevOps Tools.
- Education may be listed under Education, Background, degree, or university text.

If the answer is not available in the context, say exactly:
"{FALLBACK_ANSWER}"
Use this fallback only when the context truly does not contain the answer.

Answer concisely and cite the page number like this: [page 3].

Context:
{context}

Question:
{question}

Answer:"""


class LocalLLM:
    def __init__(self) -> None:
        from app.config import settings

        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL

    def generate_text(self, prompt: str, timeout: int = 120) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def generate_answer(self, question: str, context_chunks: list[dict]) -> dict:
        import requests

        prompt = build_rag_prompt(question, context_chunks)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("response", "").strip() or FALLBACK_ANSWER
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        generated_tokens = int(payload.get("eval_count") or 0)
        return {
            "answer": answer,
            "usage": {
                "total_duration_ms": _ns_to_ms(payload.get("total_duration")),
                "load_duration_ms": _ns_to_ms(payload.get("load_duration")),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
                "prompt_eval_duration_ms": _ns_to_ms(payload.get("prompt_eval_duration")),
                "eval_duration_ms": _ns_to_ms(payload.get("eval_duration")),
            },
        }
