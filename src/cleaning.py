def clean_text(text: str) -> str:
    """
    Remove known noise patterns from extracted PDF text.
    """
   
    noise ='Download from finelybook www.finelybook.com'  

   

    text = text.replace("\x00", '')
    text = text.replace(noise, "")

    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    return text




def clean_documents(documents):
    """
    Clean the text of each document in the list.

    Args:
        documents (list): A list of documents to be cleaned.

    Returns:
        list: A list of cleaned documents.
    """
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents












if __name__ == "__main__":

    from loaders import load_pdf

    documents = load_pdf("C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/raw/Hands On Machine Learning.pdf")

    cleaned_documents = clean_documents(documents)

    print("Number of documents cleaned:", len(cleaned_documents))
    print("---")
    print(repr(cleaned_documents[0].page_content))