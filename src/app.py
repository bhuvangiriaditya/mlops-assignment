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
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
        return response

app.add_middleware(LoggingMiddleware)

@app.get("/metrics")
def metrics():
    # Simple mockup metrics
    return {
        "status": "up", 
        "request_count_total": 42, # Mock
        "latency_seconds_sum": 1.2 # Mock
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
