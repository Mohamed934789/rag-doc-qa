from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Chunk the documents into smaller pieces.

    Args:
        documents (list): A list of documents to be chunked.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        
    )

    chunked_documents = text_splitter.split_documents(documents)

    return chunked_documents



def generate_chunk_ids(chunked_documents):
    page_chunk_counts = {}  # هيتبع عدد الـ chunks لكل صفحة
    
    for chunk in chunked_documents:
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        
        page_chunk_counts[str(source)+str(page)]=page_chunk_counts.get(str(source)+str(page), 0) + 1
        
        chunk.metadata["chunk_id"] = f"{source}_page{page}_chunk{page_chunk_counts[str(source)+str(page)]}"
        
    
    return chunked_documents


if __name__ == "__main__":

    from loaders import load_pdf
    documents = load_pdf("C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/raw/Hands On Machine Learning.pdf")
    chunked_documents = chunk_documents(documents)

    print("Number of chunks:", len(chunked_documents))
    print("---")
    print("First chunk content:", chunked_documents[0].page_content)
    print("---")
    print("First chunk metadata:", chunked_documents[0].metadata)