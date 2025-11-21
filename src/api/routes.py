from fastapi import APIRouter, HTTPException
from src.api.schemas import SearchRequest, SearchResponse
from src.core.search_engine import SearchEngine

router = APIRouter()

# Initialize SearchEngine once (singleton-ish)
# In a real app, use dependency injection or lifespan events
search_engine = SearchEngine()

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        results = search_engine.search(request.query, k=request.k, alpha=request.alpha)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint returning index stats.
    """
    try:
        doc_count = len(search_engine.documents) if search_engine.documents else 0
        index_size = search_engine.index.ntotal if search_engine.index else 0
        return {
            "status": "healthy",
            "documents_indexed": doc_count,
            "vector_index_size": index_size,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
