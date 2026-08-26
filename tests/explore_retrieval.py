import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))



from retriever import retrieve_relevant_chunks
from langchain_chroma import Chroma
from embeddings import get_embedding_function

vectorstore = Chroma(
    persist_directory="C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/vectorstore",
    embedding_function=get_embedding_function()
)

for q in ["Why does my model perform great on training data but badly on new data"]:
    print("Query:", q)
    chunks = retrieve_relevant_chunks(vectorstore, q, k=4)
    for c in chunks:
        print("-", c.page_content[:150].replace("\n", " "))
    print("===")