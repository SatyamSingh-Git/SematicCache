# SemantiCache:An Intelligent, Caching-Enabled Vector Search Engine

# Project Design Document

## 1\. Feature Specification

This project is divided into **Core Requirements** (from the assignment) and **Strategic Enhancements** (to improve quality and demonstrate expertise).

| Category | Core Requirement (Assignment) | Strategic Enhancements (Bonus) |
| :--- | :--- | :--- |
| **Data Processing** | [cite\_start]Load `.txt` files, clean text, remove HTML [cite: 22-28]. | **Smart Chunking:** Split long docs into 256-token windows with overlap to prevent truncation. |
| **Embedding** | [cite\_start]`all-MiniLM-L6-v2` model[cite: 38]. | [cite\_start]**Batch Processing:** Generate embeddings in batches for speed[cite: 103]. |
| **Caching** | JSON/SQLite cache. [cite\_start]Recompute only if file hash changes[cite: 41, 54]. | **Vector DB (Chroma/Qdrant):** Use a persistent vector store instead of a simple JSON file for scalability. |
| **Indexing** | [cite\_start]FAISS Vector Search[cite: 58]. | **Hybrid Search:** Combine FAISS results with BM25 (keyword) results for higher accuracy. |
| **Retrieval** | [cite\_start]Top-k results based on Cosine Similarity[cite: 64]. | **Re-Ranking:** Pass top 20 results through a Cross-Encoder to refine the order based on deep semantic matching. |
| **API** | [cite\_start]FastAPI endpoint returning results + scores [cite: 72-80]. | **Telemetry:** Add logging for query latency and cache hit rates. |
| **Explanation** | [cite\_start]Show "Why this" & Overlap Ratio [cite: 95-97]. | **Keyword Highlighting:** Programmatically mark matching terms in the preview snippet. |
| **Deployment** | [cite\_start]GitHub Repo + README[cite: 106]. | **Dockerization:** Full `docker-compose` setup for one-command startup. |

-----

## 2\. System Architecture

[Image of RAG system architecture]

The data flow will operate in two distinct pipelines:

1.  **Ingestion Pipeline (Offline/Startup):**
    `Raw Text` $\rightarrow$ `Cleaning` $\rightarrow$ `Chunking` $\rightarrow$ `Hashing` $\rightarrow$ `Cache Check` $\rightarrow$ `Embedding Model` $\rightarrow$ `FAISS Index`.

2.  **Retrieval Pipeline (Online/API):**
    `User Query` $\rightarrow$ `Embedding` $\rightarrow$ `Hybrid Search (Vector + Keyword)` $\rightarrow$ `Re-Ranking` $\rightarrow$ `Result Formatting` $\rightarrow$ `JSON Response`.

-----

## 3\. Detailed Development Plan


### Phase 1: Environment & Data Setup

**Goal:** Set up the repo and get raw data ready.

1.  **Initialize Repo:** Create the folder structure (src, data, docs).
2.  **Virtual Env:** Create `requirements.txt` containing `sentence-transformers`, `faiss-cpu`, `fastapi`, `uvicorn`, `numpy`, `scikit-learn` (for fetching data).
3.  [cite\_start]**Download Data:** Write a script to download the "20 Newsgroups" dataset [cite: 13] or use a directory of manual text files.
      * [cite\_start]*Tip:* Filter for just 100-200 files initially as requested[cite: 11].

### Phase 2: The "Embedder" & "Chunker" (Core Logic)

**Goal:** Process text into vectors with caching.

1.  [cite\_start]**Implement Cleaning:** Create `clean_text(text)` to lowercase and remove extra spaces/HTML [cite: 23-28].
2.  **Implement Chunking (Enhancement):** Instead of embedding the whole file, write a `chunk_text(text, size=256, overlap=50)` function.
3.  **Implement Caching:**
      * Before embedding, calculate `SHA256` of the chunk text.
      * Check your Cache (SQLite or JSON).
      * *Logic:* `IF hash in cache -> load embedding. [cite_start]ELSE -> generate -> save to cache.`[cite: 54].
4.  [cite\_start]**Generate Embeddings:** Use `sentence-transformers` to convert text chunks to vectors[cite: 38].

### Phase 3: The Search Index

**Goal:** Make the vectors searchable.

1.  [cite\_start]**Build FAISS Index:** Feed your vectors into `faiss.IndexFlatIP` (Inner Product is equivalent to Cosine Similarity for normalized vectors)[cite: 60].
2.  **Save/Load Index:** Ensure the index can be saved to disk (`index.bin`) so you don't rebuild it every time you restart the server.
3.  **Search Function:** Write a function that takes a query vector and returns `top_k` indices and distances.

### Phase 4: The API (FastAPI)

**Goal:** Expose the logic via HTTP.

1.  **Setup FastAPI:** Initialize the app.
2.  **Endpoint `/search`:**
      * [cite\_start]Accepts `{"text": "query", "top_k": 5}`[cite: 76].
      * [cite\_start]Runs the embedding generation on the query[cite: 78].
      * Queries FAISS.
      * [cite\_start]**Ranking Logic:** Calculate the "Overlap Ratio" (intersection of query words vs document words) as requested[cite: 97].
      * **Enhancement:** If implementing Re-ranking, add the Cross-Encoder step here to re-sort the top 10 results.
3.  [cite\_start]**Response:** Structure the JSON exactly as requested (doc\_id, score, preview) [cite: 83-90].

### Phase 5: The User Interface (Bonus)

**Goal:** Visual demo.

1.  **Streamlit App:** Create a simple `app.py`.
2.  Input text box for query.
3.  Display results in cards with the "Score" and "Explanation" visible.

### Phase 6: Docker & Documentation

**Goal:** Professional delivery.

1.  **Dockerfile:** Python base image $\rightarrow$ Install requirements $\rightarrow$ Copy src $\rightarrow$ Run API.
2.  [cite\_start]**README:** strictly follow the assignment list: Design choices, Folder structure, How caching works [cite: 111-116].

-----

## 4\. Recommended Directory Structure

[cite\_start]This structure is modular [cite: 65] and separates concerns effectively.

```text
search-engine-assignment/
[cite_start]├── data/                   # Ignored by Git [cite: 108]
│   ├── raw_docs/           # The .txt files
│   ├── cache/              # SQLite db or JSON cache files
│   └── indices/            # Saved FAISS index
[cite_start]├── src/                    # Source code [cite: 107]
│   ├── __init__.py
│   ├── config.py           # Paths and constants
│   ├── preprocessing.py    # Cleaning and Chunking logic
[cite_start]│   ├── embedder.py         # Model loading and embedding generation [cite: 66]
[cite_start]│   ├── cache_manager.py    # Logic to check/update cache [cite: 67]
[cite_start]│   ├── search_engine.py    # FAISS and Hybrid search logic [cite: 68]
│   ├── ranker.py           # (Enhancement) Cross-encoder re-ranking
[cite_start]│   └── api.py              # FastAPI endpoints [cite: 69]
[cite_start]├── app_ui.py               # Streamlit frontend (Bonus) [cite: 100]
├── Dockerfile              # (Enhancement) Containerization
├── docker-compose.yml      # (Enhancement) Orchestration
[cite_start]├── requirements.txt        # Dependencies [cite: 110]
[cite_start]└── README.md               # Documentation [cite: 109]
```

