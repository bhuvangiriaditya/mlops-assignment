# MLOps Assignment 2: Cats vs Dogs Classification

End-to-end MLOps pipeline for binary image classification (`cat` vs `dog`) with:
- model development and experiment tracking
- API packaging and containerization
- CI for test/build/smoke validation
- manual CD for image publish and Kubernetes deployment
- basic runtime logging and metrics

## 1. Project Overview

Use case: pet adoption platform image classifier.  
Dataset: Kaggle Cats and Dogs dataset.  
Input preprocessing: `224x224` RGB images with augmentation for train split.  
Data split: `80/10/10` train/validation/test.

## 2. Repository Structure

```text
.
├── src/
│   ├── app.py           # FastAPI inference service (+health, +predict, +metrics)
│   ├── infer.py         # ModelService prediction wrapper
│   ├── train.py         # Model training + MLflow logging
│   ├── dataset.py       # Data loading, split, transforms
│   └── model.py         # Baseline CNN
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_dataset.py
├── scripts/
│   ├── setup_data.sh
│   └── smoke_test.sh
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yaml
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── model.pth
```

## 3. Local Setup

### Prerequisites
- Python 3.9+
- Docker
- (Optional) Kubernetes cluster (`kind`, `minikube`, etc.)
- Kaggle API token for dataset download

### Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download dataset
```bash
chmod +x scripts/setup_data.sh
./scripts/setup_data.sh
```

### DVC-tracked data
The dataset folders are tracked with DVC metadata:
- `data/raw.dvc`
- `data/subset.dvc`
- `data/processed.dvc`

After changing data, update metadata:
```bash
dvc add data/raw data/subset data/processed
git add data/raw.dvc data/subset.dvc data/processed.dvc .gitignore
```

### Train model (MLflow tracked)
```bash
python -m src.train --epochs 5 --batch_size 32 --lr 0.001
```

Start MLflow UI:
```bash
mlflow ui --port 5000
```

### Run inference API
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `POST /predict` (multipart file upload)
- `GET /metrics`

Example prediction request:
```bash
curl -X POST -F "file=@data/subset/cats/1.jpg" http://localhost:8000/predict
```

## 4. Testing

Run unit tests:
```bash
pytest
```

Smoke test:
```bash
chmod +x scripts/smoke_test.sh
./scripts/smoke_test.sh
```

## 5. Docker

Build image:
```bash
docker build -t cats-dogs-classifier:latest .
```

Run container:
```bash
docker run --rm -p 8000:8000 cats-dogs-classifier:latest
```

Or with Compose:
```bash
docker compose up --build
```

## 6. Kubernetes Deployment

Apply manifests:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Port-forward service:
```bash
kubectl port-forward svc/cats-dogs-classifier-svc 8000:80
```

## 7. CI/CD

### CI (`.github/workflows/ci.yml`)
Triggered on push/PR to `main`:
1. install dependencies
2. run tests
3. build Docker image
4. run container and execute smoke test
5. on push to `main`, push image to Docker Hub `aditya3298/mlops-2` with `${sha}` and `latest` tags
6. upload CI artifacts: JUnit report, coverage XML/HTML, smoke test log, container log

### CD (`.github/workflows/cd.yaml`)
Triggered automatically after `MLOps CI` completes successfully on `main`:
1. pick `aditya3298/mlops-2:latest`
2. deploy/update Kubernetes workload (if `KUBE_CONFIG_DATA` secret is set)
3. run post-deploy smoke test

Required secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `KUBE_CONFIG_DATA` (plain kubeconfig content) for deploy job

## 8. Monitoring and Logging

Implemented in `src/app.py`:
- request logging middleware (method, path, status, latency)
- in-app metrics:
  - `request_count_total`
  - `latency_seconds_sum`
  - `latency_seconds_avg`
