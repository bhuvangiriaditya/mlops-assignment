from fastapi.testclient import TestClient
import src.app as app_module

client = TestClient(app_module.app)


def _reset_metrics_state():
    with app_module.metrics_lock:
        app_module.request_count_total = 0
        app_module.latency_seconds_sum = 0.0
        app_module.request_count_other = 0
        for endpoint in app_module.request_count_by_endpoint:
            app_module.request_count_by_endpoint[endpoint] = 0

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


def test_metrics_breakdown():
    _reset_metrics_state()

    client.get("/health")
    client.get("/metrics")
    files = {"file": ("test.jpg", b"fake image data", "image/jpeg")}
    client.post("/predict", files=files)
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    breakdown = payload["request_count_breakdown"]

    assert breakdown["/health"] >= 1
    assert breakdown["/predict"] >= 1
    assert breakdown["/metrics"] >= 1
    assert payload["request_count_total"] == (
        breakdown["/health"] + breakdown["/metrics"] + breakdown["/predict"] + breakdown["other"]
    )


def test_predict_with_loaded_service(monkeypatch):
    class StubService:
        def predict(self, _image_bytes):
            return {"label": "dog", "confidence": 0.99}

    monkeypatch.setattr(app_module, "service", StubService())
    files = {"file": ("test.jpg", b"fake image data", "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.json()["label"] == "dog"
    assert response.json()["confidence"] == 0.99
