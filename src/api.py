"""Bounded multipart PDF uploads; no caller-controlled server paths."""
import tempfile
from pathlib import Path, PureWindowsPath
from threading import Lock
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from config import settings
from src.pipeline import RAGPipeline

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

def create_app(pipeline=None, upload_dir=None, max_upload_bytes=None):
    app = FastAPI(title="RAG Document Q&A")
    instance = pipeline
    init_lock = Lock()
    maximum = settings.MAX_UPLOAD_BYTES if max_upload_bytes is None else max_upload_bytes
    def get_pipeline():
        nonlocal instance
        with init_lock:
            if instance is None:
                instance = RAGPipeline()
        return instance

    @app.get("/health")
    def health():
        return {"status": "ok", "corpus_scope": "single-owner local workspace"}

    @app.post("/ask")
    def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(422, "Question must not be blank")
        try:
            return {"question": request.question, **get_pipeline().ask(request.question)}
        except LookupError:
            raise HTTPException(409, "Index a PDF first") from None
        except Exception:
            raise HTTPException(503, "Answer generation unavailable; check local configuration") from None

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        name = PureWindowsPath(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            await file.close()
            raise HTTPException(415, "A PDF file is required")
        directory = Path(upload_dir or settings.DATA_RAW_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=directory) as work:
                path = Path(work) / "upload.pdf"
                size = 0
                with path.open("wb") as target:
                    while data := await file.read(65536):
                        size += len(data)
                        if size > maximum:
                            raise HTTPException(413, "PDF exceeds upload limit")
                        target.write(data)
                with path.open("rb") as source:
                    if source.read(5) != b"%PDF-":
                        raise HTTPException(415, "Invalid PDF header")
                try:
                    result = await run_in_threadpool(get_pipeline().build_vectorstore, path, name)
                except ValueError:
                    raise HTTPException(422, "PDF cannot be indexed; use a readable, unencrypted PDF") from None
                except Exception:
                    raise HTTPException(503, "Indexing unavailable; check local configuration") from None
                return {"message": "Document indexed", **result}
        finally:
            await file.close()
    return app

app = create_app()
