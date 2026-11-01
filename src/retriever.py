def retrieve_relevant_chunks(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)
