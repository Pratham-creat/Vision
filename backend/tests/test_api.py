from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


def test_health_and_upload_validation():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/api/v1/images/upload", files={"file": ("bad.txt", BytesIO(b"no"), "text/plain")})
        assert response.status_code == 415


def test_upload_and_process():
    output = BytesIO()
    Image.new("RGB", (10, 8), (120, 80, 50)).save(output, "PNG")
    with TestClient(app) as client:
        uploaded = client.post("/api/v1/images/upload", files={"file": ("sample.png", output.getvalue(), "image/png")})
        assert uploaded.status_code == 200
        image_id = uploaded.json()["image_id"]
        processed = client.post("/api/v1/images/process", json={"image_id": image_id, "method": "rgb", "budget": 2, "seed": 1})
        assert processed.status_code == 200
        assert processed.json()["metrics"]["ssim"] <= 1
        assert client.get(f"/api/v1/images/{image_id}/download?variant=modified").status_code == 200
