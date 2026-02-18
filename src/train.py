import argparse
import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SimpleCNN
from src.dataset import get_dataloaders
from sklearn.metrics import accuracy_score

def train(epochs=5, batch_size=32, lr=0.001):
    mlflow.set_experiment("cats_vs_dogs")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        
        # Data
        data_dir = "data/subset" if os.path.exists("data/subset/cats") else "data/raw"
        print(f"Using data from: {data_dir}")
        train_loader, val_loader, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
        if train_loader is None:
            print("Error: Dataset not found or empty.")
            return

        # Model
        device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
        model = SimpleCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"Training on {device}...")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs) > 0.5
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = accuracy_score(all_labels, all_preds)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Val Acc: {val_acc:.4f}")
            
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Test Evaluation
        print("Evaluating on Test Set...")
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                
        test_acc = accuracy_score(test_labels, test_preds)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
