#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
TMP_DIR="${TMP_DIR:-/tmp/mlops-smoke}"

echo "Waiting for service (up to 60s)..."
healthy=false
for _ in {1..12}; do
    if curl -fsS "${BASE_URL}/health" | grep -q '"status":"healthy"'; then
        healthy=true
        break
    fi
    echo "Waiting..."
    sleep 5
done

if [[ "${healthy}" != "true" ]]; then
    echo "Service health check failed."
    exit 1
fi

echo "Testing prediction..."
TEST_IMG="$(find data/subset/cats data/raw/PetImages/Cat -name '*.jpg' -print -quit 2>/dev/null || true)"

if [[ -z "${TEST_IMG}" ]]; then
    mkdir -p "${TMP_DIR}"
    TEST_IMG="${TMP_DIR}/synthetic_test.jpg"
    TMP_DIR="${TMP_DIR}" python - <<'PY'
from PIL import Image
import os

img = Image.new("RGB", (224, 224), color=(128, 96, 64))
out_path = os.path.join(os.environ["TMP_DIR"], "synthetic_test.jpg")
img.save(out_path, format="JPEG")
PY
fi

response="$(curl -fsS -X POST -F "file=@${TEST_IMG}" "${BASE_URL}/predict")"
echo "Prediction response: ${response}"

SMOKE_RESPONSE="${response}" python - <<'PY'
import json
import os

payload = json.loads(os.environ["SMOKE_RESPONSE"])
if "label" not in payload or "confidence" not in payload:
    raise SystemExit("Prediction payload missing label/confidence.")
if not isinstance(payload["label"], str):
    raise SystemExit("label should be a string.")
if not isinstance(payload["confidence"], (float, int)):
    raise SystemExit("confidence should be numeric.")
PY

echo "Smoke test passed!"
