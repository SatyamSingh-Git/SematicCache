import faiss
import numpy as np
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.config import INDICES_DIR, EMBEDDING_DIMENSION
from src.core.embedder import Embedder
from src.core.query_cache import SemanticQueryCache

class SearchEngine:
    def __init__(self):
        self.embedder = Embedder()
        self.query_cache = SemanticQueryCache()
        # Load CrossEncoder for re-ranking
        print("Loading CrossEncoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("CrossEncoder loaded.")
        
        self.documents = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        self._load_data()
        self._build_indices()

    def _load_data(self):
        # Load metadata
        meta_path = INDICES_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.documents = json.load(f)
        else:
            print("Warning: metadata.json not found. Run ingest.py first.")
        
        # Load embeddings
        emb_path = INDICES_DIR / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        else:
            print("Warning: embeddings.npy not found. Run ingest.py first.")

    def _build_indices(self):
        if self.embeddings is not None:
            # Normalize for Cosine Similarity
            faiss.normalize_L2(self.embeddings)
            # FAISS Index (Inner Product)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.index.add(self.embeddings)
            print(f"FAISS index built with {self.index.ntotal} vectors (Cosine Similarity).")

        if self.documents:
            # BM25 Index
            # Simple tokenization by splitting on space
            tokenized_corpus = [doc["content"].lower().split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("BM25 index built.")

    def _calculate_overlap(self, query: str, doc_content: str) -> tuple[float, list[str]]:
        """
        Calculates the percentage of query terms present in the document and returns the terms.
        """
        q_terms = set(query.lower().split())
        if not q_terms:
            return 0.0, []
        doc_terms = set(doc_content.lower().split())
        intersection = q_terms.intersection(doc_terms)
        overlap = len(intersection)
        return overlap / len(q_terms), list(intersection)

    def search(self, query: str, k: int = 5, alpha: float = 0.5, rerank: bool = True):
        """
        Hybrid search using FAISS + BM25 with optional Re-ranking.
        alpha: Weight for vector search (0.0 to 1.0).
        rerank: Whether to apply Cross-Encoder re-ranking.
        """
        if not self.documents or not self.index:
            return []

        # 1. Vector Search
        query_embedding = self.embedder.embed_documents([query])[0]
        # Normalize query for Cosine Similarity
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Check Semantic Cache
        cached_results = self.query_cache.check(query_embedding)
        if cached_results:
            return cached_results[:k]

        # FAISS expects 2D array
        # Fetch more candidates for re-ranking (e.g., 20 or 2*k)
        initial_k = 20 if rerank else k * 2
        D, I = self.index.search(np.array([query_embedding]), initial_k) 
        
        vector_results = {}
        for dist, idx in zip(D[0], I[0]):
            if idx != -1:
                # For Inner Product with normalized vectors, dist is cosine similarity (-1 to 1)
                score = float(dist) 
                vector_results[idx] = score

        # 2. BM25 Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:initial_k]
        
        bm25_results = {}
        max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        if max_bm25 == 0: max_bm25 = 1.0
        
        for idx in top_bm25_indices:
            bm25_results[idx] = bm25_scores[idx] / max_bm25

        # 3. Merge Scores
        all_indices = set(vector_results.keys()) | set(bm25_results.keys())
        candidates = []
        
        for idx in all_indices:
            v_score = vector_results.get(idx, 0.0)
            b_score = bm25_results.get(idx, 0.0)
            final_score = (alpha * v_score) + ((1 - alpha) * b_score)
            
            overlap_score, matched_keywords = self._calculate_overlap(query, self.documents[idx]["content"])
            
            # Generate explanation
            explanation = f"Matched with {overlap_score:.0%} keyword overlap ({', '.join(matched_keywords)})."
            if v_score > 0.5:
                explanation += f" High semantic similarity ({v_score:.2f})."
            
            candidates.append({
                "id": int(idx),
                "score": final_score,
                "filename": self.documents[idx]["filename"],
                "content": self.documents[idx]["content"],
                "vector_score": v_score,
                "bm25_score": b_score,
                "overlap_score": overlap_score,
                "matched_keywords": matched_keywords,
                "explanation": explanation
            })
            
        # Sort by initial hybrid score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 4. Re-ranking
        if rerank and candidates:
            # Take top 20 candidates for re-ranking
            top_candidates = candidates[:20]
            cross_inp = [[query, c["content"]] for c in top_candidates]
            cross_scores = self.cross_encoder.predict(cross_inp)
            
            for i, score in enumerate(cross_scores):
                top_candidates[i]["score"] = float(score) # Replace score with cross-encoder score
                top_candidates[i]["rerank_score"] = float(score)
            
            # Sort by new score
            top_candidates.sort(key=lambda x: x["score"], reverse=True)
            final_results = top_candidates[:k]
            self.query_cache.add(query, query_embedding, final_results)
            return final_results
            
        final_results = candidates[:k]
        self.query_cache.add(query, query_embedding, final_results)
        return final_results

if __name__ == "__main__":
    engine = SearchEngine()
    results = engine.search("artificial intelligence", k=3)
    for res in results:
        print(f"[{res['score']:.4f}] {res['filename']}")
