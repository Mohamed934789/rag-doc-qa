<div align="center">

# 📚 RAG Document Q&A

### Ask questions about your documents. Get grounded, source-backed answers.

A production-style **Retrieval-Augmented Generation (RAG)** system built from scratch — from raw PDF ingestion to a containerized, deployable API.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1F2937?style=for-the-badge&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=databricks&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## 🎯 What This Is

You give it a PDF. You ask it a question. It searches the document's actual content, pulls the relevant passages, and generates an answer **grounded strictly in that content** — refusing to hallucinate when the answer isn't there.

Built end-to-end by hand: no black-box templates, no copy-pasted boilerplate. Every stage — ingestion, cleaning, chunking, embedding, retrieval, generation, evaluation, the API, and the container — was designed, implemented, debugged, and tested deliberately.

<div align="center">

```
📄 PDF  →  🧹 Clean  →  ✂️ Chunk  →  🧠 Embed  →  🗄️ Store
                                                      │
❓ Question  →  🔍 Retrieve  →  📝 Prompt  →  🤖 LLM  →  ✅ Answer + Sources
```

</div>

---

## ✨ Key Features

| | |
|---|---|
| 🧩 **Modular Pipeline** | Each stage (load, clean, chunk, embed, store, retrieve, generate) lives in its own module — no monolithic scripts |
| 🛡️ **Grounded Answers** | The LLM is explicitly instructed to answer *only* from retrieved context — and to say "I don't know" otherwise, preventing hallucination |
| 🔁 **Idempotent Indexing** | Documents are chunked into deterministic, content-based IDs — re-uploading the same file never creates duplicate vectors |
| 🧽 **Robust Text Cleaning** | Handles real-world PDF extraction issues: null bytes, broken Unicode surrogate pairs, embedded artifacts |
| ⚠️ **Production Error Handling** | Meaningful exceptions with preserved stack traces (`raise ... from e`), not silent failures |
| 🔄 **Automatic Retries** | Transient OpenAI API failures are retried with exponential backoff |
| 📊 **Structured Logging** | Centralized logging config — every module logs with timestamp, source, and level |
| ⚙️ **Centralized Config** | All tunables (chunk size, models, paths) live in one place, not scattered magic numbers |
| ✅ **Evaluation Harness** | Automated retrieval-quality checks against a curated question set |
| 🌐 **REST API** | Clean FastAPI interface (`/upload`, `/ask`) with auto-generated interactive docs |
| 🐳 **Containerized** | Fully dockerized — runs identically on any machine with zero local setup |

---

## 🏗️ Architecture

<div align="center">

```
┌──────────────────────────────────────────────────────────┐
│                     🐳 Docker Container                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  🌐 FastAPI Layer                    │  │
│  │            POST /upload      POST /ask               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │              🧠 RAGPipeline                     │  │  │
│  │  │                                                  │  │  │
│  │  │   load_pdf → clean → chunk → generate_ids       │  │  │
│  │  │            → embed → store                      │  │  │
│  │  │                                                  │  │  │
│  │  │   retrieve → build_prompt → generate_answer     │  │  │
│  │  │  ┌────────────────────────────────────────┐    │  │  │
│  │  │  │         🗄️  ChromaDB (vector store)      │    │  │  │
│  │  │  └────────────────────────────────────────┘    │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
                    🔑 OpenAI API
              (embeddings + chat completion)
```

</div>

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Why |
|---|---|---|
| 🤖 LLM & Embeddings | **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`) | Strong quality-to-cost ratio for a portfolio-scale project |
| 🔗 Orchestration | **LangChain** | Industry-standard abstractions for loaders, splitters, and chains |
| 🗄️ Vector Store | **ChromaDB** | Lightweight, local, zero-infra setup — ideal for this scale |
| 🌐 API Framework | **FastAPI** | Async-ready, automatic validation via Pydantic, self-documenting |
| 🐳 Containerization | **Docker** | Reproducible, portable deployment across any machine |
| 🐍 Language | **Python 3.11** | — |

</div>

---

## 📁 Project Structure

```
rag-doc-qa/
├── 🐳 Dockerfile
├── 🚫 .dockerignore
├── 🚫 .gitignore
├── 🔑 .env                    # API keys (not committed)
├── 📋 requirements.txt
│
├── ⚙️  config/
│   ├── __init__.py
│   └── settings.py            # centralized config: paths, models, chunk size, logging
│
├── 🧠 src/
│   ├── api.py                 # FastAPI app — /upload, /ask
│   ├── pipeline.py            # RAGPipeline — orchestrates the full flow
│   ├── loaders.py             # PDF loading
│   ├── cleaning.py            # text cleaning (encoding fixes, noise removal)
│   ├── chunking.py            # text splitting + deterministic chunk IDs
│   ├── embeddings.py          # embedding model config
│   ├── vectorstore.py         # Chroma create/upsert logic (idempotent)
│   ├── retriever.py           # similarity search
│   └── generation.py          # prompt construction + LLM call
│
├── 🧪 tests/
│   ├── explore_retrieval.py   # manual retrieval exploration
│   └── evaluate_retrieval.py  # automated retrieval evaluation
│
└── 📂 data/
    ├── raw/                   # source PDFs
    └── vectorstore/           # persisted Chroma DB
```

---

## 🚀 Getting Started

### Option 1 — Run with Docker (recommended)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd rag-doc-qa

# 2. Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Build the image
docker build -t rag-doc-qa .

# 4. Run the container
docker run -p 8000:8000 --env-file .env rag-doc-qa
```

Then open **http://127.0.0.1:8000/docs** for the interactive API playground. 🎉

### Option 2 — Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# create a .env with OPENAI_API_KEY=...

uvicorn src.api:app --reload
```

---

## 📡 API Usage

### `POST /upload` — Index a document

```json
{
  "file_path": "/app/data/raw/your-document.pdf"
}
```

### `POST /ask` — Ask a question

```json
{
  "question": "What is regularization?"
}
```

**Response:**

```json
{
  "question": "What is regularization?",
  "answer": "Regularization is a technique used to reduce the risk of overfitting by adding a penalty for overly complex model weights..."
}
```

If the answer isn't in the indexed documents, the system responds honestly:

```json
{
  "question": "What is the capital of France?",
  "answer": "I don't know."
}
```

---

## 🧠 Design Decisions Worth Highlighting

> A few engineering choices that came from hitting real problems, not just following a tutorial.

- **Deterministic chunk IDs** (`source + page + chunk_index`) so re-indexing the same document is idempotent — no duplicate vectors, ever.
- **Two-stage text cleaning**: stripped both PDF encoding artifacts (null bytes, broken Unicode surrogate pairs) *before* chunking — cleaning after chunking would corrupt chunk boundaries.
- **Grounded prompting**: the LLM is instructed to answer only from retrieved context, verified empirically — out-of-scope questions reliably return "I don't know" instead of hallucinated answers.
- **Retrieval evaluated before optimizing**: techniques like MMR, reranking, and multi-query retrieval were deliberately **not** added without evidence they were needed — validated through a manual + automated evaluation pass first.

---

## 🗺️ Roadmap

- [ ] File upload via `multipart/form-data` instead of server-side file paths
- [ ] Multi-query retrieval / reranking (if evaluation shows a need)
- [ ] LLM-as-judge for answer-quality scoring
- [ ] `docker-compose.yml` for multi-service orchestration

---

<div align="center">

Built as a hands-on deep dive into production RAG engineering — from first principles to a deployable container.

</div>
