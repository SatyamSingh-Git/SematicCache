import json
import numpy as np
from typing import List, Dict, Optional
from src.config import QUERY_CACHE_FILE, QUERY_CACHE_THRESHOLD

class SemanticQueryCache:
    """
    Caches search results based on semantic similarity of queries.
    If a user asks a question similar to a previous one, return cached results.
    """
    def __init__(self, cache_path=QUERY_CACHE_FILE, threshold=QUERY_CACHE_THRESHOLD):
        self.cache_path = cache_path
        self.threshold = threshold
        self.cache = self._load_cache()

    def _load_cache(self) -> List[Dict]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Assumes vectors are already normalized (which they are from the embedder usually)
        # But to be safe:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def check(self, query_embedding: np.ndarray) -> Optional[List[Dict]]:
        """
        Checks if a semantically similar query exists in the cache.
        Returns the cached results if found, else None.
        """
        best_score = -1
        best_results = None
        
        for entry in self.cache:
            cached_embedding = np.array(entry["embedding"])
            score = self._cosine_similarity(query_embedding, cached_embedding)
            
            if score > best_score:
                best_score = score
                best_results = entry["results"]

        if best_score >= self.threshold:
            print(f"⚡ Semantic Cache HIT! (Score: {best_score:.4f})")
            return best_results
        
        return None

    def add(self, query_text: str, query_embedding: np.ndarray, results: List[Dict]):
        """
        Adds a new query and its results to the cache.
        """
        # Avoid growing indefinitely - simple FIFO or limit could be added here
        # For now, just append
        entry = {
            "query": query_text,
            "embedding": query_embedding.tolist(),
            "results": results
        }
        self.cache.append(entry)
        self._save_cache()
