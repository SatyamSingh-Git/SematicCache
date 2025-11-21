from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        Returns a numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
