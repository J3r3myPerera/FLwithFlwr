import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Improved Multi-layer Neural Network for Savings Potential Classification.
    
    Improvements:
    - Added weight initialization (Xavier/He)
    - Added optional weight decay support
    - Configurable dropout
    - LeakyReLU for better gradient flow
    """
    
    def __init__(self, num_features: int, num_classes: int = 3, dropout: float = 0.3):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(num_features, 128)  # Increased from 64
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)  # Increased from 32
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, 32)  # Increased from 16
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout * 0.5)  # Less dropout in later layers
        
        self.fc4 = nn.Linear(32, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)  # LeakyReLU for better gradients
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x


def train(model: nn.Module, trainloader, optimizer, epochs: int, device, 
          weight_decay: float = 0.0001):
    """
    Train the MLP model with optional gradient clipping.
    
    Improvements:
    - Gradient clipping to prevent exploding gradients
    - Label smoothing for better generalization
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


def test(model: nn.Module, testloader, device) -> tuple:
    """Evaluate the model on the test/validation set."""
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
