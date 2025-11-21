import sys
import numpy as np
from src.core.preprocessing import TextLoader
from src.core.embedder import Embedder
from src.core.cache_manager import CacheManager
from src.config import INDICES_DIR

def main():
    # 1. Load Documents
    loader = TextLoader()
    documents = loader.load_files()
    
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2. Initialize Cache and Embedder
    cache_manager = CacheManager()
    
    # 3. Check Cache
    print("Checking cache...")
    
    # We need to adapt how we interact with CacheManager because we now have chunks.
    # We will use a unique identifier for each chunk as the "filename" in the cache.
    
    docs_to_embed = []
    final_embeddings = [None] * len(documents)
    indices_to_embed = []

    for i, doc in enumerate(documents):
        # Create a unique ID for the chunk
        chunk_unique_id = f"{doc['filename']}_chunk_{doc['chunk_id']}"
        
        # Check cache using the unique ID
        emb = cache_manager.get_embedding(chunk_unique_id, doc["content"])
        
        if emb is not None:
            final_embeddings[i] = emb
        else:
            docs_to_embed.append(doc)
            indices_to_embed.append(i)
    
    if docs_to_embed:
        print(f"Found {len(docs_to_embed)} new or modified chunks.")
        embedder = Embedder()
        
        # 4. Embed New Documents
        texts = [doc["content"] for doc in docs_to_embed]
        new_embeddings = embedder.embed_documents(texts)
        
        # 5. Update Cache and Final Embeddings List
        for i, idx in enumerate(indices_to_embed):
            doc = docs_to_embed[i]
            emb = new_embeddings[i]
            
            chunk_unique_id = f"{doc['filename']}_chunk_{doc['chunk_id']}"
            cache_manager.update_entry(chunk_unique_id, doc["content"], emb)
            
            final_embeddings[idx] = emb
            
        cache_manager.save_cache()
    else:
        print("All chunks are already cached. No new embeddings needed.")

    # Ensure all embeddings are present and valid
    # Filter out any Nones if something went wrong (though logic above should handle it)
    valid_embeddings = [e for e in final_embeddings if e is not None]
    
    if len(valid_embeddings) != len(documents):
        print("Error: Mismatch in number of embeddings and documents.")
        return

    # Convert to numpy array for FAISS (Phase 3)
    embeddings_array = np.array(valid_embeddings).astype('float32')
    print(f"Total Embeddings Shape: {embeddings_array.shape}")
    
    # Save embeddings array for FAISS indexing later
    np.save(INDICES_DIR / "embeddings.npy", embeddings_array)
    print(f"Embeddings saved to {INDICES_DIR / 'embeddings.npy'}")

    # Save metadata to ensure alignment
    import json
    metadata = [{"filename": doc["filename"], "path": doc["path"], "content": doc["content"]} for doc in documents]
    with open(INDICES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {INDICES_DIR / 'metadata.json'}")

if __name__ == "__main__":
    main()
