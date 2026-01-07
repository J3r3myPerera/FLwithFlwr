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


def train(net, trainloader, optimizer, epochs, device: str, max_grad_norm: float = 1.0):
    """Train the MLP on the training set (FedAvg style).
    
    Args:
        net: Neural network model
        trainloader: Training data loader
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            # Clip gradients to prevent exploding gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()


def train_fedprox(net, trainloader, optimizer, epochs, device: str, 
                  global_params: list, proximal_mu: float, max_grad_norm: float = 1.0):
    """Train with FedProx proximal term.
    
    FedProx adds a proximal term to the loss to prevent client drift:
    L_total = L_CE + (mu/2) * ||w - w_global||^2
    
    This keeps local updates close to the global model, which is especially
    beneficial in non-IID settings.
    
    Args:
        net: Neural network model
        trainloader: Training data loader
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        global_params: List of global model parameters (numpy arrays)
        proximal_mu: Proximal term coefficient (higher = stronger regularization)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
    """
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    
    # Convert global params to tensors and store them
    global_tensors = [torch.tensor(p, device=device, dtype=torch.float32) 
                      for p in global_params]
    
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Cross-entropy loss
            ce_loss = criterion(net(features), labels)
            
            # Proximal term: (mu/2) * ||w - w_global||^2
            proximal_term = 0.0
            for local_param, global_param in zip(net.parameters(), global_tensors):
                proximal_term += torch.sum((local_param - global_param) ** 2)
            proximal_term = (proximal_mu / 2.0) * proximal_term
            
            # Total loss
            loss = ce_loss + proximal_term
            
            loss.backward()
            # Clip gradients to prevent exploding gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
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
