import os
import sys
import pytest
from fastapi.testclient import TestClient

root_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_project_path)

try:
    from main import app
except ImportError as e:
    print("Error importing FastAPI app: {e}")
    print("Make sure main.py is in the root directory.")
    app = None

if app:
    client = TestClient(app)
else:
    client = None


@pytest.mark.skipif(client is None, reason='Couldn\'t connect to FastAPI')
def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the RAG API!"}

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Database not available in CI environment")
@pytest.mark.skipif(client is None, reason='Couldn\'t connect to FastAPI')
def test_ask_question():
    response = client.post('/ask', json={"question": "What is Dijkstra's Algorithm?"})
    assert response.status_code == 200
    response_data = response.json()
    assert "answer" in response_data
    assert isinstance(response_data["answer"], str)
    assert len(response_data["answer"]) > 0

@pytest.mark.skipif(client is None, reason='Couldn\'t connec tot FastAPI')
def test_non_existing_path():
    response = client.get('/nonexistingpath')
    assert response.status_code == 404
