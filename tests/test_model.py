import torch
import pytest
from src.model import SimpleCNN

def test_model_structure():
    model = SimpleCNN()
    assert isinstance(model, torch.nn.Module)
    
def test_model_forward():
    model = SimpleCNN()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    # Output should be (1, 1) logits
    assert output.shape == (1, 1)
