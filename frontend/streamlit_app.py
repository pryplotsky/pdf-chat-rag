import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def fetch_documents() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/documents", timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_document_chunks(document_id: str) -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/documents/{document_id}/chunks", timeout=20)
    response.raise_for_status()
    return response.json()


def upload_pdf(uploaded_file) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }
    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def ask_question(document_id: str, question: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/documents/{document_id}/chat",
        json={"question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def fetch_analytics_summary() -> dict:
    response = requests.get(f"{BACKEND_URL}/analytics/summary", timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_chat_metrics() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/analytics/chats", timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_ingestion_metrics() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/analytics/ingestion", timeout=20)
    response.raise_for_status()
    return response.json()


def seconds(ms: float | int | None) -> str:
    return f"{float(ms or 0) / 1000:.2f}s"


def number(value: float | int | None) -> str:
    return f"{float(value or 0):.0f}"


st.set_page_config(page_title="PDF Chat RAG", layout="wide")
st.title("PDF Chat RAG")
st.warning("Demo app. Please do not upload confidential documents.")

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

with st.sidebar:
    st.header("Documents")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is not None and st.button("Upload", use_container_width=True):
        try:
            result = upload_pdf(uploaded_pdf)
            st.session_state.active_document_id = result["document_id"]
            st.session_state.last_answer = None
            st.success("Uploaded")
            st.rerun()
        except requests.HTTPError as exc:
            detail = exc.response.json().get("detail", str(exc))
            st.error(detail)
        except requests.RequestException as exc:
            st.error(f"Backend request failed: {exc}")

try:
    documents = fetch_documents()
except requests.RequestException as exc:
    documents = []
    st.sidebar.error(f"Backend unavailable: {exc}")

if documents:
    labels = {
        f"{doc['filename']} ({doc['id'][:8]})": doc["id"] for doc in documents
    }
    current_index = 0
    if st.session_state.active_document_id in labels.values():
        current_index = list(labels.values()).index(st.session_state.active_document_id)
    selected_label = st.sidebar.selectbox(
        "Active document",
        options=list(labels.keys()),
        index=current_index,
    )
    st.session_state.active_document_id = labels[selected_label]
else:
    st.sidebar.info("No documents uploaded")

active_document = next(
    (doc for doc in documents if doc["id"] == st.session_state.active_document_id),
    None,
)

chat_tab, documents_tab, analytics_tab = st.tabs(["Chat", "Documents", "Analytics"])

with chat_tab:
    if active_document is None:
        st.info("Upload a PDF to begin.")
    else:
        st.subheader(active_document["filename"])
        st.caption(
            f"{active_document['num_pages']} pages | "
            f"{active_document['num_chunks']} chunks | "
            f"{active_document['id']}"
        )

        with st.form("chat-form", clear_on_submit=False):
            question = st.text_input("Question", placeholder="What is this document about?")
            submitted = st.form_submit_button("Ask", use_container_width=False)

        if submitted:
            if not question.strip():
                st.error("Enter a question.")
            else:
                with st.spinner("Asking local model..."):
                    try:
                        st.session_state.last_answer = ask_question(
                            active_document["id"],
                            question.strip(),
                        )
                    except requests.HTTPError as exc:
                        detail = exc.response.json().get("detail", str(exc))
                        st.error(detail)
                    except requests.RequestException as exc:
                        st.error(f"Backend request failed: {exc}")

        if st.session_state.last_answer:
            st.markdown("### Answer")
            st.write(st.session_state.last_answer["answer"])

            metrics = st.session_state.last_answer.get("metrics")
            if metrics:
                metric_cols = st.columns(4)
                metric_cols[0].metric("Response", seconds(metrics["total_response_time_ms"]))
                metric_cols[1].metric("LLM", seconds(metrics["llm_generation_time_ms"]))
                metric_cols[2].metric("Prompt Tokens", number(metrics["prompt_tokens"]))
                metric_cols[3].metric("Generated", number(metrics["generated_tokens"]))

                with st.expander("Timing breakdown", expanded=False):
                    timing_cols = st.columns(4)
                    timing_cols[0].metric(
                        "Query Enhancement",
                        seconds(metrics.get("query_enhancement_time_ms")),
                    )
                    timing_cols[1].metric(
                        "Query Embedding",
                        seconds(metrics.get("query_embedding_time_ms")),
                    )
                    timing_cols[2].metric("Qdrant", seconds(metrics.get("qdrant_search_time_ms")))
                    timing_cols[3].metric("BM25", seconds(metrics.get("bm25_search_time_ms")))

                    timing_cols = st.columns(4)
                    timing_cols[0].metric("Fusion", seconds(metrics.get("fusion_time_ms")))
                    timing_cols[1].metric("Rerank", seconds(metrics.get("rerank_time_ms")))
                    timing_cols[2].metric(
                        "Prompt Build",
                        seconds(metrics.get("prompt_build_time_ms")),
                    )
                    timing_cols[3].metric("LLM", seconds(metrics.get("llm_generation_time_ms")))

            enhanced_queries = st.session_state.last_answer.get("enhanced_queries", [])
            if enhanced_queries:
                with st.expander("Enhanced queries", expanded=False):
                    for index, enhanced_query in enumerate(enhanced_queries, start=1):
                        st.markdown(f"{index}. {enhanced_query}")

            st.markdown("### Sources")
            for index, source in enumerate(st.session_state.last_answer["sources"], start=1):
                label = (
                    f"Source {index} | page {source['page_number']} | "
                    f"relevance {source['score']:.3f}"
                )
                with st.expander(label):
                    st.write(source["quote"])

with documents_tab:
    st.subheader("Documents")
    if documents:
        st.dataframe(documents, use_container_width=True, hide_index=True)
        if active_document is not None:
            st.markdown("### Chunks")
            try:
                chunks = fetch_document_chunks(active_document["id"])
                if chunks:
                    st.caption(f"{len(chunks)} chunks for {active_document['filename']}")
                    st.dataframe(chunks, use_container_width=True, hide_index=True)
                else:
                    st.info("No chunks found for this document.")
            except requests.RequestException as exc:
                st.error(f"Could not load chunks: {exc}")
    else:
        st.info("No documents uploaded.")

with analytics_tab:
    st.subheader("Analytics")
    try:
        summary = fetch_analytics_summary()
        chat_metrics = fetch_chat_metrics()
        ingestion_metrics = fetch_ingestion_metrics()

        row_one = st.columns(4)
        row_one[0].metric("Average Response", seconds(summary["avg_response_time_ms"]))
        row_one[1].metric("Average Preprocessing", seconds(summary["avg_preprocessing_time_ms"]))
        row_one[2].metric("Average LLM", seconds(summary["avg_llm_generation_time_ms"]))
        row_one[3].metric("Total Questions", number(summary["total_questions"]))

        row_two = st.columns(4)
        row_two[0].metric("Average Prompt Tokens", number(summary["avg_prompt_tokens"]))
        row_two[1].metric("Average Generated Tokens", number(summary["avg_generated_tokens"]))
        row_two[2].metric("Average Retrieved Chunks", number(summary["avg_retrieved_chunks"]))
        row_two[3].metric("Total Documents", number(summary["total_documents"]))

        row_three = st.columns(4)
        row_three[0].metric(
            "Average Query Enhancement",
            seconds(summary.get("avg_query_enhancement_time_ms")),
        )
        row_three[1].metric("Average Qdrant", seconds(summary.get("avg_qdrant_search_time_ms")))
        row_three[2].metric("Average BM25", seconds(summary.get("avg_bm25_search_time_ms")))
        row_three[3].metric("Average Rerank", seconds(summary.get("avg_rerank_time_ms")))

        st.markdown("### Recent Questions")
        if chat_metrics:
            st.dataframe(chat_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("No chat metrics yet.")

        st.markdown("### Document Preprocessing")
        if ingestion_metrics:
            st.dataframe(ingestion_metrics, use_container_width=True, hide_index=True)
        else:
            st.info("No ingestion metrics yet.")
    except requests.RequestException as exc:
        st.error(f"Analytics unavailable: {exc}")
