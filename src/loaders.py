from config import settings
from langchain_community.document_loaders import PyPDFLoader
import logging

logger = logging.getLogger(__name__)

def load_pdf(file_path: str):
    """
    Load a PDF file and return its content as a list of pages.

    Args:
        file_path (str): The path to the PDF file."""

    logger.info(f"Loading PDF from: {file_path}")
    try:
        loader= PyPDFLoader(file_path)

        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages")
        return documents
    except ValueError as e:

        if "not a valid file" in str(e):
             logger.error(f"File not found or invalid: {file_path}")
             raise ValueError(f"File not found or invalid: {file_path}") from e
        raise  # أي ValueError تاني، سيبه يتصعّد زي ما هو من غير ما نغير رسالته



if __name__ == "__main__":
    documents = load_pdf("C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/raw/Hands On Machine Learning.pdf")
    
    # print("Number of documents:", len(documents))
    # print("---")
    # print("Metadata of first page:", documents[0].metadata)
    # print("---")
    # print("Content of first page (first 300 chars):", documents[0].page_content[:300])
    print("Length of fifth page content:", len(documents[5].page_content))
