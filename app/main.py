from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph import ask_question

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Ask questions about the Agentic AI eBook.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    text: str
    metadata: dict
    relevance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: float


@app.get("/")
def home():
    return {"message": "Agentic AI RAG Chatbot API is running."}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        result = ask_question(request.question)
        return AnswerResponse(
            answer=result["answer"],
            sources=result["context"],
            confidence=result["score"],
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error