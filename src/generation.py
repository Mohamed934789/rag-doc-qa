from langchain_openai import ChatOpenAI

def build_prompt(context: str, question: str) -> str:
    """
    Build a prompt that instructs the LLM to answer only from the given context.
    """
    prompt = f"""You are a helpful assistant. Use the following pieces of context to answer the question at the end.
      if the context doesn't contain the answer, just say you don't know and don't try to make up an answer.


Context: {context}

Question: {question}

Answer:"""

    
    return prompt


def generate_answer(prompt: str, model_name: str = "gpt-4o-mini"):
    """
    Generate an answer using the specified LLM model.
    """
    llm = ChatOpenAI(model_name=model_name)

    response = llm.invoke(prompt)
    
    return response.content



if __name__ == "__main__":
    from retriever import retrieve_relevant_chunks
    from vectorstore import Chroma
    from embeddings import get_embedding_function
    
   
    vectorstore= Chroma(
        persist_directory="C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/vectorstore",
        embedding_function=get_embedding_function()
    )

    question = "What is the difference between supervised and unsupervised learning?"

    relevant_chunks = retrieve_relevant_chunks(vectorstore, question)

    context ="\n\n".join([chunk.page_content for chunk in relevant_chunks])

    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)

    print("Question:", question)
    print("---")
    print("Answer:", answer)