import time
import threading
import uvicorn
import requests
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_search_api():
    response = client.post(
        "/api/v1/search",
        json={"query": "artificial intelligence", "k": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    print("\nAPI Test Passed!")
    print(data)

if __name__ == "__main__":
    test_search_api()
