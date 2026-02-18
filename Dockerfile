FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Copy application code
COPY src/ src/
COPY model.pth .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
