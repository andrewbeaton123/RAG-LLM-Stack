
from fastapi import APIRouter

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


@router.get("/generate")
def generate():
    
    return {"text": "Test Response !"}
