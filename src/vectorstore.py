from langchain_chroma import Chroma

def open_vectorstore(embedding_function, persist_directory):
    return Chroma(collection_name="documents", persist_directory=str(persist_directory), embedding_function=embedding_function)

def create_vectorstore(chunks, embedding_function, persist_directory, store=None):
    store = store if store is not None else open_vectorstore(embedding_function, persist_directory)
    if not chunks:
        raise ValueError("No indexable chunks")
    document_id = chunks[0].metadata["document_id"]
    previous = set(store.get(where={"document_id": document_id})["ids"])
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    # Embed/add before removing obsolete IDs, so an embedding failure retains old evidence.
    store.add_documents(chunks, ids=ids)
    obsolete = previous - set(ids)
    if obsolete:
        store.delete(ids=sorted(obsolete))
    return store
