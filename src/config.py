import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
INDICES_DIR = DATA_DIR / "indices"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Query Cache Configuration
QUERY_CACHE_FILE = CACHE_DIR / "query_cache.json"
QUERY_CACHE_THRESHOLD = 0.85 # Similarity threshold for semantic cache hit
