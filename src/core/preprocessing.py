import os
import re
from typing import List, Dict
from pathlib import Path
from src.config import RAW_DATA_DIR
from sklearn.datasets import fetch_20newsgroups

class TextLoader:
    """
    Handles loading and cleaning of text files from the raw data directory.
    """
    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = data_dir

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: removes extra whitespace, newlines, etc.
        """
        # Remove HTML tags (if any)
        text = re.sub(r'<[^>]+>', '', text)
        # Replace multiple newlines/tabs with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, window_size: int = 256, overlap: int = 50) -> List[str]:
        """
        Splits text into chunks using a sliding window strategy.
        """
        words = text.split()
        if len(words) <= window_size:
            return [text]
        
        chunks = []
        step = window_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + window_size])
            chunks.append(chunk)
        return chunks

    def load_files(self) -> List[Dict[str, str]]:
        """
        Loads all .txt files from the data directory and chunks them.
        Returns a list of dictionaries with 'filename', 'content', 'path', 'chunk_id'.
        """
        documents = []
        if not self.data_dir.exists():
            print(f"Directory {self.data_dir} does not exist. Creating it...")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check if directory is empty, if so, download sample data
        if not any(self.data_dir.glob("*.txt")):
            print("No text files found. Downloading 20 Newsgroups dataset for testing...")
            self.download_20newsgroups()

        for file_path in self.data_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    cleaned_content = self.clean_text(content)
                    if cleaned_content:
                        chunks = self.chunk_text(cleaned_content)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                "filename": file_path.name,
                                "content": chunk,
                                "path": str(file_path),
                                "chunk_id": i,
                                "original_filename": file_path.name # Keep track of parent
                            })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} chunks from {self.data_dir}")
        return documents

    def download_20newsgroups(self, limit=200):
        """
        Downloads the 20 Newsgroups dataset and saves a subset as .txt files.
        """
        try:
            print(f"Fetching 20 Newsgroups dataset (limit={limit})...")
            dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            
            count = 0
            for i, text in enumerate(dataset.data):
                if count >= limit:
                    break
                
                clean_content = self.clean_text(text)
                if len(clean_content) < 50: # Skip very short texts
                    continue
                    
                filename = f"newsgroup_{i}.txt"
                file_path = self.data_dir / filename
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text) # Write original text, let load_files handle cleaning/chunking
                count += 1
            
            print(f"Successfully saved {count} documents to {self.data_dir}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    # Test the loader
    loader = TextLoader()
    docs = loader.load_files()
    for doc in docs[:3]:
        print(f"File: {doc['filename']}")
        print(f"Content Preview: {doc['content'][:100]}...")
