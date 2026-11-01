"""Remove invalid encoding characters without altering meaningful whitespace."""
def clean_text(text):
    return text.replace("\x00", "").encode("utf-8", errors="ignore").decode("utf-8")

def clean_documents(documents):
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents
