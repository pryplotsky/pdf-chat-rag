import re

import fitz


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be processed for ingestion."""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_pages(file_path: str, max_pages: int) -> list[dict]:
    if max_pages <= 0:
        raise PDFProcessingError("MAX_PAGES must be greater than zero.")

    try:
        with fitz.open(file_path) as document:
            if document.page_count > max_pages:
                raise PDFProcessingError(
                    f"PDF has {document.page_count} pages; the limit is {max_pages}."
                )

            pages: list[dict] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = _clean_text(page.get_text("text"))
                if not text:
                    continue
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )

            return pages
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Could not read PDF: {exc}") from exc
