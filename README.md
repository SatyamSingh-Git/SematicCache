# 🧠 SemanticCache

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-2.3-orange.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**A High-Performance, Intelligent Semantic Search Engine with Advanced Caching**

[Report Bug](https://github.com/yourusername/SemanticCache/issues) · [Request Feature](https://github.com/yourusername/SemanticCache/issues)

</div>

---

**SemanticCache** is designed to demonstrate advanced information retrieval techniques. It combines the precision of keyword search with the understanding of vector search, enhanced by a semantic caching layer that drastically reduces latency for similar queries.

---

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Architecture](#-architecture)
- [Flow Diagram](#-flow-diagram)
- [How Caching Works](#-how-caching-works)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Embedding Generation](#running-embedding-generation)
  - [Starting the API](#starting-the-api)
  - [Launching the UI](#launching-the-ui)
- [Folder Structure](#-folder-structure)
- [Design Choices](#-design-choices)
- [Features](#-features)
- [Advanced Capabilities](#-advanced-capabilities)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🚀 About the Project

SemanticCache was built to solve the problem of redundant computation in search systems. Traditional search engines re-process every query, even if it's semantically identical to a previous one (e.g., "How to install?" vs "Installation guide"). 

This project implements a **Semantic Query Cache** that understands the *meaning* of a query. If a user asks a question that is semantically similar to a cached query, the system returns the cached results instantly, bypassing the expensive search and re-ranking pipeline.

---

## 🏗 Architecture

The system is built on a modular architecture separating ingestion, core search logic, and presentation layers.

1.  **Ingestion Layer**: Reads raw text files, chunks them using a sliding window approach, and generates embeddings using `sentence-transformers`.
2.  **Storage Layer**: 
    *   **Vector Index**: Numpy-based storage for dense vector embeddings.
    *   **Metadata Store**: JSON-based storage for document content and file mapping.
    *   **Cache Store**: JSON-based persistence for Embedding Cache and Query Cache.
3.  **Core Engine**:
    *   **Hybrid Search**: Combines **BM25** (keyword) and **Cosine Similarity** (vector) scores.
    *   **Re-ranker**: Uses a Cross-Encoder model to refine the top-k results.
4.  **Interface Layer**:
    *   **FastAPI**: RESTful endpoints for programmatic access.
    *   **Streamlit**: Interactive dashboard for users.

---

## 🔄 Flow Diagram

```mermaid
graph TD
    User[User Query] --> UI[Streamlit UI / API]
    UI --> CacheCheck{Query Cache Hit?}
    
    CacheCheck -- Yes (Similarity > 0.9) --> ReturnCache[Return Cached Results]
    
    CacheCheck -- No --> Embed[Generate Query Embedding]
    Embed --> VectorSearch[Vector Search (FAISS/Numpy)]
    Embed --> KeywordSearch[BM25 Search]
    
    VectorSearch --> Merge[Merge & Normalize Scores]
    KeywordSearch --> Merge
    
    Merge --> Rerank[Cross-Encoder Re-ranking]
    Rerank --> UpdateCache[Update Query Cache]
    UpdateCache --> ReturnNew[Return Fresh Results]
    
    ReturnCache --> Display[Display Results]
    ReturnNew --> Display
```

---

## ⚡ How Caching Works

### 1. Embedding Cache (Ingestion Phase)
To avoid re-computing embeddings for files that haven't changed:
*   The system calculates a hash of the file content + chunk configuration.
*   Before generating an embedding, it checks `embeddings_cache.json`.
*   **Benefit**: Drastically speeds up the `ingest.py` process on subsequent runs.

### 2. Semantic Query Cache (Search Phase)
To optimize response times for users:
*   Incoming queries are embedded into a vector.
*   The system calculates the **Cosine Similarity** between the new query vector and stored query vectors in `query_cache.json`.
*   If a similarity score exceeds the threshold (default: `0.9`), the cached results are returned.
*   **Benefit**: "What is the capital of France?" and "Capital city of France" are treated as the same query, saving compute resources.

---

## 🏁 Getting Started

### Prerequisites
*   Python 3.10+
*   Virtual Environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SemanticCache.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Embedding Generation
Before searching, you must ingest your data. Place `.txt` files in `data/raw/`.
```bash
python ingest.py
```
*This will chunk documents, generate embeddings, and save indices to `data/indices/`.*

### Starting the API
To run the backend server:
```bash
python src/api/main.py
# OR
uvicorn src.api.main:app --reload
```
*Access Swagger docs at `http://localhost:8000/docs`*

### Launching the UI
To explore the data visually:
```bash
streamlit run src/ui/streamlit_app.py
```

---

## 📂 Folder Structure

```text
SemanticCache/
├── data/
│   ├── cache/          # Stores embedding_cache.json and query_cache.json
│   ├── indices/        # Stores embeddings.npy and metadata.json
│   └── raw/            # Input text files
├── Docs/               # Project documentation
├── src/
│   ├── api/            # FastAPI routes and schemas
│   ├── core/           # Core logic (Search, Embedder, CacheManager)
│   ├── ui/             # Streamlit dashboard
│   └── config.py       # Global configuration
├── ingest.py           # Script to process documents
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🎨 Design Choices

1.  **Hybrid Search**: Pure vector search can miss specific keywords (e.g., acronyms), while keyword search misses context. We combine both using a weighted average (Alpha parameter) to get the best of both worlds.
2.  **Sliding Window Chunking**: Instead of hard cuts, we use overlapping chunks to ensure context isn't lost at the boundaries of a split.
3.  **Local-First**: The project uses file-based storage (Numpy/JSON) instead of a heavy vector database (like Pinecone/Milvus) to ensure it's easy to run locally without infrastructure overhead.

---

## ✨ Features

*   **🔍 Hybrid Search**: Adjustable weight between Keyword (BM25) and Vector search.
*   **💡 Smart Explanations**: Provides context on why a result was selected.
*   **🖍️ Syntax Highlighting**: Automatically highlights search terms in the results.
*   **📊 Analytics Dashboard**: Visualizes score distributions and result metrics.
*   **☁️ Word Cloud**: Generates word clouds from search results for quick topic overview.
*   **📥 Export Data**: Download search results as CSV or JSON.
*   **👁️ Document Preview**: View the full content of the source document directly in the UI.

---

## 🧠 Advanced Features

### Semantic Query Caching
The system doesn't just look for exact string matches. It uses a vector-based threshold to determine if a new query is "close enough" to a previous one.
*   *Threshold*: `0.9` (Configurable)
*   *Model*: `all-MiniLM-L6-v2`

### Cross-Encoder Re-ranking
After retrieving the top candidates using the bi-encoder (fast), we pass the top results through a Cross-Encoder (more accurate but slower). This model looks at the query and document *together* to output a final relevance score, significantly improving precision.

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact: theindianboy555@gmail.com
