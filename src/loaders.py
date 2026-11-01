"""Native PDF ingestion; image-only PDFs require external OCR."""
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError("PDF does not exist")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")
    documents = [Document(page_content=page.extract_text() or "", metadata={"source": path.name, "page": i})
                 for i, page in enumerate(reader.pages)]
    if not any(d.page_content.strip() for d in documents):
        raise ValueError("No native PDF text found; OCR is not implemented")
    return documents
