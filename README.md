# RAG Document Q&A

A single-owner PDF question-answering prototype using LangChain, Chroma, OpenAI and FastAPI.
It implements native-text PDF ingestion, content-sensitive chunk IDs, persistent retrieval,
and answers with returned PDF-page excerpts. This is a local prototype, not a multi-tenant service.

## Setup
Python 3.12 is the tested target.
```bash
git clone https://github.com/Mohamed934789/rag-doc-qa.git
cd rag-doc-qa
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Copy .env.example to .env.local and set your own OpenAI key for live use.
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs. `/health` requires no provider credentials.

```bash
curl -F "file=@your-document.pdf" http://127.0.0.1:8000/upload
curl -H "Content-Type: application/json" -d '{"question":"What does the document say?"}' http://127.0.0.1:8000/ask
```
`/upload` accepts multipart PDF bytes, up to 10 MiB, never server file paths. Re-uploading
the same basename replaces that document's chunks; use distinct names for distinct documents.
Answers return `question`, `answer`, and `sources` with one-based PDF pages, chunk IDs and quotes.
Indexing and answering use paid APIs with your configured account. No provider call is needed for tests.

## Architecture and limits
`loaders → cleaning → chunking → embeddings → Chroma → retrieval → generation → API`.
The pipeline is injectable for offline tests; configuration is in `config/settings.py`.
Storage is project-relative by default and can be overridden with `RAG_VECTORSTORE_DIR`.
The corpus is shared by local requests. Bind to localhost; authentication and tenant separation
are not implemented. Image-only/encrypted PDFs are rejected; OCR is not implemented.
Prompted abstention and returned excerpts are not a guarantee of faithful answers.

## Verification
```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
Tests exercise real local Chroma persistence with deterministic fixture embeddings, reindex/shrink,
page citations, multipart validation, safe errors and restart behavior. They do not measure live LLM
accuracy. The original retrieval exploration scripts are retained as historical experiments;
use `python -m tests.evaluate_retrieval` only after setting up live credentials and a corpus.

## Data and evaluation
Use your own authorized PDFs or generate a synthetic sample with `python scripts/make_sample.py`.
An older committed textbook PDF is retained pending an owner redistribution review; do not treat
its presence as a license grant. No benchmark accuracy is claimed. Offline fixture retrieval is a
regression test, not a representative semantic-quality evaluation.

The optional `tests/evaluate_retrieval.py` uses your configured indexed corpus and embedding provider.
Its textbook keyword checks are a limited retrieval smoke check, not a grounded-answer benchmark;
they are excluded from offline CI. Both evaluation scripts are portable and have no calls on import.
