from langchain_openai import OpenAIEmbeddings


def get_embedding_function():
    """
    Get the OpenAI embeddings model.

    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")  