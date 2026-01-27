
from fastapi import FastAPI
from llm_service.app.api.llm_api import router as llm_router
app = FastAPI()

@app.get("/health")
def health(): 
    return {"status":"ok"}


app.include_router(llm_router)