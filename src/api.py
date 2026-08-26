from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import RAGPipeline

# 1) إنشاء الـ app والـ pipeline مرة واحدة بس، وقت ما السيرفر بيبدأ
app = FastAPI(title="RAG Document Q&A API")

pipeline = RAGPipeline()

import os
from langchain_chroma import Chroma

if os.path.exists(pipeline.persist_directory) and os.listdir(pipeline.persist_directory):
    pipeline.vectorstore = Chroma(
        persist_directory=pipeline.persist_directory,
        embedding_function=pipeline.embedding_function
    )


# 2) شكل البيانات المتوقعة لكل endpoint
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


class UploadRequest(BaseModel):
    file_path: str


# 3) الـ endpoints
@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not hasattr(pipeline, "vectorstore"):
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document first via /upload."
        )

    answer = pipeline.ask(request.question)
    return AskResponse(question=request.question, answer=answer)


@app.post("/upload")
def upload(request: UploadRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    pipeline.build_vectorstore(request.file_path)
    return {"message": f"Document processed and indexed: {request.file_path}"}