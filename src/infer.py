import torch
from torchvision import transforms
from PIL import Image
from src.model import SimpleCNN
import io

class ModelService:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cpu')
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()
            
        label = "dog" if prob > 0.5 else "cat"
        confidence = prob if prob > 0.5 else 1 - prob
        return {"label": label, "confidence": confidence}
