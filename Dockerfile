FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
# Pin torch cpu version to reduce image size
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY model.pth .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
