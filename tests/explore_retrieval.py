"""Optional provider-backed retrieval inspection; import has no network effects."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import RAGPipeline

if __name__ == "__main__":
    query=input("Question: ").strip()
    for chunk in RAGPipeline().retrieve(query):
        print(chunk.metadata.get("source"), chunk.metadata.get("page"), chunk.page_content[:150])
