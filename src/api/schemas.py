from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    alpha: float = 0.5

class SearchResult(BaseModel):
    id: int
    score: float
    filename: str
    content: str
    vector_score: float
    bm25_score: float
    overlap_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
