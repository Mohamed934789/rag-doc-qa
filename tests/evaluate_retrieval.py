import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from embeddings import get_embedding_function
from langchain_chroma import Chroma

from retriever import retrieve_relevant_chunks


evaluation_set = [
    {
        "question": "What is regularization?",
        "expected_keyword": "regulariz"  # يغطي regularization, regularize, regularized
    },
    {
        "question": "What is a neural network?",
        "expected_keyword": "neural network"
    },
    {
        "question": "What is overfitting?",
        "expected_keyword": "overfit"
    },
    {
        "question": "What is gradient descent?",
        "expected_keyword": "gradient descent"
    },
    {
        "question": "What is cross-validation?",
        "expected_keyword": "cross-validation"
    },
    {
        "question": "What is a decision tree?",
        "expected_keyword": "decision tree"
    },
    {
        "question": "What is the difference between classification and regression?",
        "expected_keyword": "classification"
    },
    {
        "question": "What is a support vector machine?",
        "expected_keyword": "support vector"
    },
]


def evaluate_retrieval(vectorstore, evaluation_set):
    results = []
    for item in evaluation_set:
        question = item["question"]
        expected_keyword = item["expected_keyword"]

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(vectorstore, question, k=4)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Check if the expected keyword is in the retrieved context
        found = expected_keyword.lower() in context.lower()
        results.append({
            "question": question,
            "expected_keyword": expected_keyword,
            "found": found,
            "context": context
        })

    passed = sum(1 for r in results if r["found"])
    total = len(results)

    print(f"\n{'='*40}")
    print(f"Evaluation Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print(f"{'='*40}\n")

    for r in results:
        status = "✅ PASS" if r["found"] else "❌ FAIL"
        print(f"{status} — {r['question']}")

vectorstore = Chroma(
    persist_directory="C:/Users/mkass/OneDrive/Desktop/AI/RAG/rag-doc-qa/data/vectorstore",
    embedding_function=get_embedding_function()
)

evaluate_retrieval(vectorstore, evaluation_set)