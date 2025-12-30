import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer Neural Network for Savings Potential Classification.
    
    Architecture:
    - Input layer: num_features (16 features from Personal Finance dataset)
    - Hidden layer 1: 64 neurons with ReLU and Dropout
    - Hidden layer 2: 32 neurons with ReLU and Dropout
    - Hidden layer 3: 16 neurons with ReLU
    - Output layer: num_classes (3 classes: Low, Medium, High)
    
    Suitable for classification with heterogeneous tabular data.
    """
    
    def __init__(self, num_features: int, num_classes: int = 3, dropout: float = 0.3):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(num_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        
        self.fc4 = nn.Linear(16, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Output layer (no activation - CrossEntropyLoss expects raw logits)
        x = self.fc4(x)
        
        return x


def train(model: nn.Module, trainloader, optimizer, epochs: int, device):
    """Train the MLP model on the training set."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Optional: print epoch stats (uncomment for debugging)
        # accuracy = 100 * correct / total
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")


def test(model: nn.Module, testloader, device) -> tuple:
    """
    Evaluate the model on the test/validation set.
    
    Returns:
        loss: Total cross-entropy loss
        accuracy: Classification accuracy (0-1)
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    return total_loss, accuracy
