import io

import torch
from PIL import Image

from src.infer import ModelService
from src.model import SimpleCNN


def _image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_model_service_predict_cat_and_dog(tmp_path):
    model_path = tmp_path / "model.pth"
    torch.save(SimpleCNN().state_dict(), model_path)

    service = ModelService(str(model_path))

    def dog_forward(_tensor):
        return torch.tensor([[10.0]])

    def cat_forward(_tensor):
        return torch.tensor([[-10.0]])

    service.model.forward = dog_forward
    dog_result = service.predict(_image_bytes())
    assert dog_result["label"] == "dog"
    assert dog_result["confidence"] > 0.5

    service.model.forward = cat_forward
    cat_result = service.predict(_image_bytes())
    assert cat_result["label"] == "cat"
    assert cat_result["confidence"] > 0.5
