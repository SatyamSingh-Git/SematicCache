from fastapi import FastAPI
from src.api.routes import router as search_router

app = FastAPI(title="SemanticCache API", version="1.0")

app.include_router(search_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to SemanticCache API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
