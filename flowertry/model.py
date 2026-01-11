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


def train_scaffold(model, trainloader, optimizer, epochs, device: str,
                   c_global: list, c_i: list, max_grad_norm: float = 1.0):
    """Train with SCAFFOLD control variates.

    SCAFFOLD client update:
    The correction term (c_global - c_i) is added to gradients during training.
    This corrects for client drift in heterogeneous settings.

    Args:
        net: Neural network model
        trainloader: Training data loader
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        c_global: Global control variate (list of numpy arrays)
        c_i: Client control variate (list of numpy arrays)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)

    Returns:
        c_i_new: Updated client control variate
        delta_c: Change in client control variate (c_i_new - c_i)
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)

    # Convert control variates to tensors
    c_global_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in c_global]
    c_i_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in c_i]

    # Store initial parameters for control variate update
    params_before = [p.detach().clone() for p in model.parameters()]

    # Count total steps for control variate calculation
    total_steps = 0

    # Training loop
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # SCAFFOLD correction: add (c_i - c_global) to gradients
            # This corrects the local gradient to align with the global objective
            with torch.no_grad():
                for param, cg, ci in zip(model.parameters(), c_global_tensors, c_i_tensors):
                    if param.grad is not None:
                        param.grad.data += (ci - cg)  # FIXED: c_i - c_global
                        # OLD (WRONG): param.grad.data += (cg - ci)

            # Clip gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            total_steps += 1

    # Compute updated client control variate
    # c_i_new = c_i - c_global + (1/(K*η)) * (x_before - x_after)
    # where K = epochs, η = learning rate
    # Simplified: c_i_new = c_global - (x_after - x_before) / (K * η)

    lr = optimizer.param_groups[0]['lr']

    c_i_new = []
    delta_c = []

    with torch.no_grad():
        for p_before, p_after, cg, ci in zip(params_before, model.parameters(),
                                              c_global_tensors, c_i_tensors):
            # Option II update: c_i_new = c_i - c_global + (x_before - x_after) / (K * η)
            # CRITICAL FIX: Use epochs, not total_steps!
            param_diff = (p_before - p_after) / (epochs * lr) if (epochs * lr) > 0 else (p_before - p_after)
            c_new = ci - cg + param_diff

            c_i_new.append(c_new.cpu().numpy())

    return c_i_new


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
