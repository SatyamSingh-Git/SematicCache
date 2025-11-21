

### 📂 Project Root: `search-engine-assignment/`

```text
search-engine-assignment/
│
├── .gitignore                  # Standard gitignore (critical: ignores /data)
[cite_start]├── README.md                   # Documentation, Setup, Architecture [cite: 109]
[cite_start]├── requirements.txt            # Python dependencies [cite: 110]
├── Dockerfile                  # Instructions to build the container
├── docker-compose.yml          # Orchestration for API + UI + Volumes
├── ingest.py                   # Script to run the one-time data processing pipeline
│
[cite_start]├── data/                       # DATA STORAGE (Ignored by Git) [cite: 108]
│   ├── raw/                    # Place your 100-200 .txt files here
│   ├── cache/                  # Stores SQLite db or JSON cache files
│   └── indices/                # Stores the .faiss vector index file
│
[cite_start]├── src/                        # SOURCE CODE [cite: 107]
│   ├── __init__.py
│   ├── config.py               # Central config (Paths, Model Names, Constants)
│   │
│   ├── core/                   # CORE LOGIC MODULES
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Text cleaning, chunking, metadata extraction
[cite_start]│   │   ├── embedder.py         # Loading model, generating embeddings [cite: 66]
[cite_start]│   │   ├── cache_manager.py    # Hashing checks, SQL/JSON read/write [cite: 67]
[cite_start]│   │   ├── search_engine.py    # FAISS index management, Hybrid search logic [cite: 68]
│   │   └── ranker.py           # (Enhancement) Cross-Encoder re-ranking logic
│   │
│   ├── api/                    # API LAYER
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app entry point
[cite_start]│   │   ├── routes.py           # The /search endpoint definition [cite: 69]
│   │   └── schemas.py          # Pydantic models for Input/Output validation
│   │
│   └── ui/                     # USER INTERFACE
[cite_start]│       └── streamlit_app.py    # Streamlit frontend code [cite: 100]
│
└── tests/                      # UNIT TESTS (Best Practice)
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_search.py
```

-----

### 📝 Key File Descriptions

Here is what goes into the specific files to handle the requirements and enhancements:

#### 1\. Root Level Files

  * **`ingest.py`**: This is the "SETUP" script. It runs the pipeline: Load text $\rightarrow$ Chunk $\rightarrow$ Embed $\rightarrow$ Cache $\rightarrow$ Build FAISS Index. You run this *once* before starting the API.
  * **`.gitignore`**:
    ```text
    __pycache__/
    *.pyc
    .env
    .DS_Store
    [cite_start]/data/* # CRITICAL: Assignment requires ignoring data [cite: 108]
    !/data/.gitkeep  # Keeps the folder structure in git even if empty
    ```

#### 2\. `src/config.py`

Holds your "Magic Numbers" so you don't hardcode them.

```python
import os

DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
CACHE_PATH = os.path.join(DATA_DIR, "cache", "embeddings.db")
INDEX_PATH = os.path.join(DATA_DIR, "indices", "vector.index")

# Model Settings
[cite_start]EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" [cite: 38]
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
```

#### 3\. `src/core/preprocessing.py`

Handles the cleaning and the **Smart Chunking enhancement**.

  * **Functions:** `clean_text(text)`, `chunk_text(text)`, `get_file_hash(filepath)`.

#### 4\. `src/core/search_engine.py`

The brain of the operation.

  * **Class `SearchEngine`**:
      * `load_index()`: Loads FAISS from disk.
      * `search(query, top_k)`: Embeds query, searches FAISS.
      * **Enhancement:** Includes the `hybrid_search` logic (BM25 + Dense) inside this class.

#### 5\. `src/api/routes.py`

Handles the request/response logic.

  * **Endpoint:** `POST /search`
  * [cite\_start]**Logic:** Receives JSON, calls `SearchEngine`, formats the output with "Why this" explanations[cite: 95], and returns the JSON response.

-----

### 🚀 How to Start Development (Using this Structure)

1.  **Create the skeleton:**
    Run these commands in your terminal to create the structure instantly:

    ```bash
    mkdir -p search-engine-assignment/data/{raw,cache,indices}
    mkdir -p search-engine-assignment/src/{core,api,ui}
    mkdir -p search-engine-assignment/tests
    touch search-engine-assignment/src/{__init__.py,config.py}
    touch search-engine-assignment/{Dockerfile,docker-compose.yml,requirements.txt,README.md,ingest.py}
    ```

2.  **Populate `requirements.txt`:**

    ```text
    sentence-transformers
    faiss-cpu
    fastapi
    uvicorn
    streamlit
    numpy
    scikit-learn
    rank_bm25
    ```
