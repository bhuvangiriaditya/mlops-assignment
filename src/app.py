from fastapi import FastAPI, UploadFile, File
from src.infer import ModelService
import os

app = FastAPI(title="Cats vs Dogs Qualifier")

# Initialize model (lazy load or on startup)
# For simplicity, we assume model artifact exists at 'mlruns/.../model.pth' 
# or we save a specific 'model.pth' after training.
# We will look for 'model.pth' in current directory for Docker convenience.

model_path = "model.pth"
service = None

@app.on_event("startup")
async def startup_event():
    global service
    if os.path.exists(model_path):
        service = ModelService(model_path)
    else:
        print("Warning: Model file not found. Inference will fail.")

# M5: Logging & Metrics
import time
import logging
from threading import Lock
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
metrics_lock = Lock()
request_count_total = 0
latency_seconds_sum = 0.0

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        global request_count_total, latency_seconds_sum
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Only count non-metrics, non-health requests
        if request.url.path not in ["/metrics", "/health"]:
            with metrics_lock:
                request_count_total += 1
                latency_seconds_sum += process_time
        
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
        return response

app.add_middleware(LoggingMiddleware)

@app.get("/metrics")
def metrics():
    with metrics_lock:
        req_count = request_count_total
        latency_sum = latency_seconds_sum
    avg_latency = (latency_sum / req_count) if req_count else 0.0
    return {
        "status": "up",
        "request_count_total": req_count,
        "latency_seconds_sum": round(latency_sum, 6),
        "latency_seconds_avg": round(avg_latency, 6),
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": service is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not service:
        return {"error": "Model not loaded"}
    
    contents = await file.read()
    result = service.predict(contents)
    logger.info(f"Prediction result: {result['label']} ({result['confidence']:.2f})")
    return result