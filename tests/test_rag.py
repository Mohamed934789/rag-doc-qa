import io
import hashlib
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from reportlab.pdfgen import canvas
from src.api import create_app
from src.pipeline import RAGPipeline

class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]
    def embed_query(self, text):
        # Deterministic offline fixture, not a semantic embedding benchmark.
        return [float(text.lower().count(w)) for w in ["orchid", "water", "sunlight", "winter", "soil", "plant"]]

def pdf(text):
    output = io.BytesIO(); c = canvas.Canvas(output)
    for i, line in enumerate(text.splitlines()): c.drawString(40, 800 - i * 16, line)
    c.save(); return output.getvalue()

def test_index_replace_citations_and_restart(tmp_path):
    path = tmp_path / "garden.pdf"
    embeddings = LocalEmbeddings()
    pipeline = RAGPipeline(persist_directory=tmp_path / "vectors", embedding_function=embeddings,
                           answer_generator=lambda prompt: "Water weekly [1].", chunk_size=90, chunk_overlap=10)
    path.write_bytes(pdf("Orchid plants need water weekly. " * 20))
    pipeline.build_vectorstore(path)
    initial = pipeline.vectorstore.get()["ids"]
    pipeline.build_vectorstore(path)
    assert set(initial) == set(pipeline.vectorstore.get()["ids"])
    path.write_bytes(pdf("Orchid plants need water weekly."))
    pipeline.build_vectorstore(path)
    assert len(pipeline.vectorstore.get()["ids"]) < len(initial)
    answer = pipeline.ask("How often should I water an orchid?")
    assert answer["sources"][0]["page"] == 1
    assert answer["sources"][0]["source"] == "garden.pdf"
    assert "Orchid" in answer["sources"][0]["quote"]
    restarted = RAGPipeline(persist_directory=tmp_path / "vectors", embedding_function=embeddings, answer_generator=lambda p: "offline")
    assert restarted.ask("orchid water")["sources"]

def test_upload_validation_and_no_server_paths(tmp_path):
    class Fake:
        def build_vectorstore(self, path, source_name):
            assert path.is_relative_to(tmp_path)
            return {"source": source_name, "chunks": 1}
        def ask(self, question): return {"answer": "offline", "sources": []}
    client = TestClient(create_app(Fake(), tmp_path, max_upload_bytes=4096))
    assert client.get("/health").status_code == 200
    assert client.post("/upload", json={"file_path": "/private/document.pdf"}).status_code == 422
    assert client.post("/upload", files={"file": ("x.pdf", b"not a pdf")}).status_code == 415
    assert client.post("/upload", files={"file": ("x.pdf", b"%PDF-" + b"x" * 4096)}).status_code == 413
    response = client.post("/upload", files={"file": ("../../garden.pdf", pdf("Orchid water"))})
    assert response.status_code == 200
    assert response.json()["source"] == "garden.pdf"
    assert not list(tmp_path.glob("*/upload.pdf"))
    assert client.post("/ask", json={"question": "  "}).status_code == 422

def test_empty_index_and_safe_error(tmp_path):
    pipeline = RAGPipeline(persist_directory=tmp_path / "vectors", embedding_function=LocalEmbeddings())
    client = TestClient(create_app(pipeline, tmp_path))
    assert client.post("/ask", json={"question": "water?"}).status_code == 409
    class Broken:
        def ask(self, question): raise RuntimeError("sensitive-value")
    response = TestClient(create_app(Broken(), tmp_path)).post("/ask", json={"question": "water?"})
    assert response.status_code == 503 and "sensitive-value" not in response.text
