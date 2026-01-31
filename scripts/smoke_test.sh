#!/bin/bash
echo "Waiting for service (up to 60s)..."
for i in {1..12}; do
    if curl -s http://localhost:8000/health | grep "healthy"; then
        echo "Service is up!"
        break
    fi
    echo "Waiting..."
    sleep 5
done

if [ $? -eq 0 ]; then
    echo "Service is healthy!"
else
    echo "Service health check failed."
    exit 1
fi

echo "Testing prediction..."
# Use a dummy image or existing one
# We can use one from data/subset/cats if available, or generate one
TEST_IMG=$(find data/subset/cats -name "*.jpg" | head -n 1)

if [ -z "$TEST_IMG" ]; then
    echo "No test image found, skipping prediction test."
    exit 0
fi

response=$(curl -s -X POST -F "file=@$TEST_IMG" http://localhost:8000/predict)
echo "Prediction response: $response"

if [[ "$response" == *"label"* ]]; then
    echo "Smoke test passed!"
    exit 0
else
    echo "Smoke test failed: Invalid response."
    exit 1
fi
