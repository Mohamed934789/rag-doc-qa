import os
from langchain_chroma import Chroma




def create_vectorstore(chunks, embedding_function, persist_directory):
    """
     Create or update a Chroma vector store from document chunks.
    """
   
    ids=[chunk.metadata["chunk_id"] for chunk in chunks]

    if os.path.exists(persist_directory) and os.listdir(persist_directory):

        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        vectorstore.add_documents(chunks, ids=ids)

    else:
        vectorstore = Chroma.from_documents(
                chunks,
                embedding=embedding_function,
                persist_directory=persist_directory,
                ids=ids
            )
  

    return vectorstore