from fastapi.testclient import TestClient
from src.app import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"
    
def test_predict_no_model():
    # Model might not be loaded in test env, so it should return error or fail nicely
    # We just want to check endpoint existence and basic handling
    # If model is not loaded, it returns {"error": "Model not loaded"}
    
    # Create dummy image
    files = {'file': ('test.jpg', b'fake image data', 'image/jpeg')}
    response = client.post("/predict", files=files)
    
    # We expect 200 OK but maybe an error message in body if svc is None
    assert response.status_code == 200
    json_resp = response.json()
    
    # If service is None (expected locally without model.pth), check for error
    if "error" in json_resp:
        assert json_resp["error"] == "Model not loaded"
