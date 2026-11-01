import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Overlap must be nonnegative and smaller than chunk size")
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(documents)

def generate_chunk_ids(chunks):
    counts = {}
    for chunk in chunks:
        key = (chunk.metadata["document_id"], chunk.metadata["page"])
        index = counts.get(key, 0)
        counts[key] = index + 1
        fingerprint = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:20]
        chunk.metadata["chunk_id"] = f"{key[0]}:{key[1]}:{index}:{fingerprint}"
    return chunks
