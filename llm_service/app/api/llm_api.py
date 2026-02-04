
from fastapi import APIRouter

from .models.generate import GenerateRequest, GenerateResponse

router = APIRouter(
    prefix = "/api/v1",
    tags=["llm"]
)

@router.get("/ping")
def ping():
    return {"message" : "pong"}


@router.get("/health")
def health():
    return {"message" : "ok"}


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    
    return {"text": "Test Response !"}
