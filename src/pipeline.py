"""Single-owner local corpus. Clients are injectable for paid-free tests."""
import hashlib
from pathlib import Path
from threading import RLock
from config import settings
from src.loaders import load_pdf
from src.cleaning import clean_documents
from src.chunking import chunk_documents, generate_chunk_ids
from src.embeddings import get_embedding_function
from src.vectorstore import open_vectorstore, create_vectorstore
from src.generation import build_prompt, generate_answer

class RAGPipeline:
    def __init__(self, chunk_size=None, chunk_overlap=None, persist_directory=None, embedding_function=None, answer_generator=None):
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.persist_directory = Path(persist_directory or settings.VECTORSTORE_DIR)
        self.embedding_function = embedding_function
        self.answer_generator = answer_generator or generate_answer
        self.vectorstore = None
        self.lock = RLock()

    def _store(self):
        if self.vectorstore is None:
            if self.embedding_function is None:
                self.embedding_function = get_embedding_function()
            self.vectorstore = open_vectorstore(self.embedding_function, self.persist_directory)
        return self.vectorstore

    def process_documents(self, file_path, source_name=None):
        source = source_name or Path(file_path).name
        document_id = hashlib.sha256(source.encode()).hexdigest()
        docs = clean_documents(load_pdf(file_path))
        for doc in docs:
            doc.metadata.update(source=source, document_id=document_id)
        return generate_chunk_ids(chunk_documents(docs, self.chunk_size, self.chunk_overlap))

    def build_vectorstore(self, file_path, source_name=None):
        chunks = self.process_documents(file_path, source_name)
        with self.lock:
            self._store()
            self.vectorstore = create_vectorstore(chunks, self.embedding_function, self.persist_directory, self.vectorstore)
        return {"source": source_name or Path(file_path).name, "chunks": len(chunks)}

    def retrieve(self, question, k=None):
        if not question.strip():
            raise ValueError("Question must not be blank")
        with self.lock:
            store = self._store()
            if not store.get(limit=1)["ids"]:
                raise LookupError("No documents indexed")
            return store.similarity_search(question, k=k or settings.DEFAULT_K)

    def ask(self, question, k=None):
        docs = self.retrieve(question, k)
        sources = [{"source": d.metadata["source"], "page": int(d.metadata["page"]) + 1,
                    "chunk_id": d.metadata["chunk_id"], "quote": d.page_content} for d in docs]
        context = "\n\n".join(f"[{i}] {d['source']}, PDF page {d['page']}:\n{d['quote']}" for i, d in enumerate(sources, 1))
        return {"answer": self.answer_generator(build_prompt(context, question)), "sources": sources}
