from config import settings

from src.loaders import load_pdf
from src.cleaning import clean_documents
from src.chunking import chunk_documents, generate_chunk_ids
from src.vectorstore import create_vectorstore
from src.embeddings import get_embedding_function
from src.retriever import retrieve_relevant_chunks
from src.generation import build_prompt, generate_answer


class RAGPipeline:
    def __init__(self,chunk_size=1000, chunk_overlap=200,persist_directory="C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/vectorstore"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.embedding_function = get_embedding_function()


    def process_documents(self, file_path: str):
        """
        Load, clean, and chunk documents from a PDF file.

        Args:
            file_path (str): The path to the PDF file."""

        # Load documents
        documents = load_pdf(file_path)

        # Clean documents
        cleaned_documents = clean_documents(documents)

        # Chunk documents
        chunked_documents = chunk_documents(cleaned_documents, self.chunk_size, self.chunk_overlap)
        
        # Generate chunk IDs
        chunked_documents_with_ids = generate_chunk_ids(chunked_documents)



        return chunked_documents_with_ids

    def build_vectorstore(self, file_path: str):

        chunks = self.process_documents(file_path)

        self.vectorstore = create_vectorstore(chunks, self.embedding_function, self.persist_directory)
        
       

        return self.vectorstore

    def ask(self, question: str, k: int = 4):
        """
        Full RAG flow: retrieve relevant chunks, build prompt, get answer.
        """
        
        relevant_chunks = retrieve_relevant_chunks(self.vectorstore, question, k)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        prompt = build_prompt(context, question)
        answer = generate_answer(prompt)

        return answer









if __name__ == "__main__":
    pipeline = RAGPipeline()

    file_path = "C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/raw/Hands On Machine Learning.pdf"
    
    
    pipeline.build_vectorstore(file_path)

    questions = [
        "What is the difference between supervised and unsupervised learning?",
        "What is AI?",
        "What is the capital of France?"
    ]

    for q in questions:
        answer = pipeline.ask(q)    
        print("Question:", q)
        print("Answer:", answer)
        print("---")