import json
import hashlib
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.config import CACHE_DIR

class CacheManager:
    def __init__(self, cache_file: str = "embeddings_cache.json"):
        self.cache_path = CACHE_DIR / cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Cache file corrupted. Starting fresh.")
                return {}
        return {}

    def save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
        print(f"Cache saved to {self.cache_path}")

    def compute_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(self, filename: str, text: str) -> Optional[np.ndarray]:
        """
        Retrieves embedding if file exists and hash matches.
        """
        current_hash = self.compute_hash(text)
        if filename in self.cache:
            entry = self.cache[filename]
            if entry["hash"] == current_hash:
                return np.array(entry["embedding"])
        return None

    def update_entry(self, filename: str, text: str, embedding: np.ndarray):
        """
        Updates the cache with new hash and embedding.
        """
        self.cache[filename] = {
            "hash": self.compute_hash(text),
            "embedding": embedding.tolist()
        }

    def filter_new_documents(self, documents: List[Dict]) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
        """
        Checks which documents need embedding.
        Returns:
        - docs_to_embed: List of documents that need embedding
        - cached_embeddings: List of embeddings (aligned with original list, None if needs embedding)
        - indices_to_embed: Indices in the original list that correspond to docs_to_embed
        """
        docs_to_embed = []
        cached_embeddings = [None] * len(documents)
        indices_to_embed = []

        for i, doc in enumerate(documents):
            emb = self.get_embedding(doc["filename"], doc["content"])
            if emb is not None:
                cached_embeddings[i] = emb
            else:
                docs_to_embed.append(doc)
                indices_to_embed.append(i)
        
        return docs_to_embed, cached_embeddings, indices_to_embed
