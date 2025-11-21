List features, final checklist for development.

### 1. Data Ingestion & Preprocessing Module
*Responsible for loading raw text and preparing it for the AI models.*

* **[Core] Text Loader:** Script to ingest 100-200 `.txt` files from a directory (or the "20 Newsgroups" dataset).
* **[Core] Text Cleaner:** Automated removal of HTML tags, extra whitespace, and special characters; conversion to lowercase.
* **[Core] Metadata Extraction:** Computation of file hash (SHA256), document length, and filename storage.
* **[Enhancement] Smart Chunking:** Instead of embedding entire documents (which get truncated), implement a **Sliding Window** strategy (e.g., 256-token chunks with 50-token overlap) to capture all context.
* **[Enhancement] Chunk Metadata:** Maintain a mapping between `chunk_id` and the original `parent_doc_id` to return the correct document during search.

### 2. Embedding & Caching Engine
*The core "brain" that converts text to numbers and manages efficiency.*

* **[Core] Transformer Model:** Integration of `sentence-transformers/all-MiniLM-L6-v2` to generate dense vector embeddings.
* **[Core] Intelligent Caching:**
    * **Storage:** SQLite database (or JSON) to store `doc_id`, `hash`, `embedding_vector`, and `timestamp`.
    * **Logic:** Before embedding, check if the file's current hash exists in the cache.
        * *Hit:* Load vector from disk (0s latency).
        * *Miss:* Generate new embedding and update cache.
* **[Enhancement] Batch Processing:** Process documents in batches (e.g., 32 chunks at a time) rather than one-by-one to maximize CPU/GPU throughput.

### 3. Search & Indexing System
*The mechanism for finding relevant results fast.*

* **[Core] Vector Index (FAISS):**
    * Implementation of `IndexFlatIP` (Inner Product) or `IndexFlatL2`.
    * Normalization of vectors to ensure Cosine Similarity accuracy.
    * Persistence: Ability to save/load the `.index` file to disk.
* **[Enhancement] Hybrid Search (Keyword + Vector):**
    * Implementation of **BM25** (using `rank_bm25`) alongside FAISS.
    * Logic to combine scores (e.g., 0.7 * Vector_Score + 0.3 * Keyword_Score) to ensure exact phrase matches (like specific acronyms) aren't missed by the semantic vector model.

### 4. Ranking & Explanation Module
*Ensures the best results appear first and explains why.*

* **[Core] Similarity Scoring:** Return results sorted by Cosine Similarity score.
* **[Core] Overlap Analysis:** Calculate and return the "Word Overlap Ratio" (percentage of query terms present in the document) as a metric.
* **[Enhancement] Cross-Encoder Re-Ranking:**
    * Take the top 20 results from the fast index.
    * Pass them through a Cross-Encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) for high-precision sorting.
* **[Enhancement] Keyword Highlighting:** In the preview text, programmatically wrap matching query terms in `<b>` tags for the UI.

### 5. API Layer (FastAPI)
*The interface for external applications to talk to your engine.*

* **[Core] Search Endpoint:** `POST /search`
    * Input: `{"query": string, "top_k": int}`.
    * Output: JSON object containing list of results, scores, and metadata.
* **[Core] Modular Architecture:** strict separation of concerns:
    * `api.py` (Routes)
    * `search_engine.py` (Logic)
    * `embedder.py` (Model)
* **[Enhancement] Telemetry/Health Check:** A `/health` or `/stats` endpoint showing the number of indexed documents and cache hit rate.

### 6. User Interface (Bonus)
*A visual way to demonstrate the project.*

* **[Core Bonus] Streamlit App:**
    * Search bar for user queries.
    * Result cards displaying:
        * Document Title
        * Relevance Score (Visual bar)
        * "Why this result?" (Overlap info)
        * Text Preview with [Enhancement] Highlighting.

### 7. DevOps & Deliverables
*Ensuring the project is reproducible and professional.*

* **[Core] Documentation:**
    * `README.md` covering:
        * Architecture Diagram.
        * "How Caching Works" (Hashing logic).
        * Setup instructions.
* **[Core] Requirements:** `requirements.txt` with pinned versions.
* **[Enhancement] Docker Support:**
    * `Dockerfile` to containerize the application.
    * `docker-compose.yml` to spin up the API and UI with a single command.
