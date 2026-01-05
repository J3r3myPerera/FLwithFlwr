import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Savings Potential Classification.
    
    Task: Classify users into 3 savings potential categories:
    - Low (<7% savings)
    - Medium (7-12% savings)
    - High (>12% savings)
    
    Input: 16 features from Indian Personal Finance dataset
    Output: 3 classes
    """
    
    def __init__(self, num_classes: int = 3) -> None:
        super(Net, self).__init__()
        
        # Input: 16 features (14 numerical + 2 categorical encoded)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str):
    """Train the MLP on the training set."""
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Evaluate the MLP on the test/validation set."""
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
