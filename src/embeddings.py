from config import settings

def get_embedding_function():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
