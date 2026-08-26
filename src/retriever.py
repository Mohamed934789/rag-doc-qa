from config import settings

def retrieve_relevant_chunks(vectorstore, query: str, k: int = 4):
    """
    Retrieve relevant document chunks from the vector store based on the query.
    """

    
    results = vectorstore.similarity_search(query, k=k)
    
    return results



if __name__ == "__main__":

    from langchain_chroma import Chroma
    from embeddings import get_embedding_function

    embedding_function = get_embedding_function()

    vectorstore = Chroma(
        persist_directory="C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/vectorstore",
        embedding_function=embedding_function
)

    query = "What is the difference between supervised and unsupervised learning?"
    relevant_chunks = retrieve_relevant_chunks(vectorstore, query)

    print("Number of relevant chunks retrieved:", len(relevant_chunks))
    print("---")
    for i, chunk in enumerate(relevant_chunks):
        print(f"Chunk {i+1} content:", chunk.page_content)
        print("Chunk metadata:", chunk.metadata)
        print("---")