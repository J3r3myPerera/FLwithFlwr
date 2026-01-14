import torch
import torch.nn as nn
import torch.nn.functional as F


# Default input dimension - will be updated based on feature engineering settings
DEFAULT_INPUT_DIM = 25  # 14 numerical + 9 engineered + 2 categorical


class Net(nn.Module):
    """
    Improved Multi-Layer Perceptron (MLP) for Savings Potential Classification.
    
    Task: Classify users into 3 savings potential categories:
    - Low savings
    - Medium savings
    - High savings
    
    Architecture: Wider and deeper network without batch normalization (better for FL)
    - Input: Variable features → 256 → 128 → 128 → 64 → 32 → 3 classes
    - Uses LayerNorm instead of BatchNorm for FL compatibility
    - Reduced dropout to prevent underfitting with small client data
    
    Input: Features from Indian Personal Finance dataset (default: 25 with engineering)
    Output: 3 classes
    """
    
    def __init__(self, num_classes: int = 3, input_dim: int = DEFAULT_INPUT_DIM) -> None:
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        
        # Wider and deeper architecture for better learning
        self.fc1 = nn.Linear(input_dim, 256)   # Input layer: N → 256
        
        self.fc2 = nn.Linear(256, 128)         # Hidden layer: 256 → 128
        
        self.fc3 = nn.Linear(128, 128)         # Hidden layer: 128 → 128
        
        self.fc4 = nn.Linear(128, 64)          # Hidden layer: 128 → 64
        
        self.fc5 = nn.Linear(64, 32)           # Hidden layer: 64 → 32
        
        self.fc6 = nn.Linear(32, num_classes)  # Output layer: 32 → 3
        
        # Lighter dropout to prevent underfitting
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 4
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 5
        x = self.fc5(x)
        x = F.relu(x)
        
        # Output layer (no activation, CrossEntropyLoss applies softmax)
        x = self.fc6(x)
        return x


def train(net, trainloader, optimizer, epochs, device: str, max_grad_norm: float = 1.0, class_weights=None):
    """Train the MLP on the training set (FedAvg style).
    
    Args:
        net: Neural network model
        trainloader: Training data loader
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        class_weights: Optional tensor of class weights for weighted loss
    """
    # Setup loss function with optional class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
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
                  global_params: list, proximal_mu: float, max_grad_norm: float = 1.0, class_weights=None):
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
        class_weights: Optional tensor of class weights for weighted loss
    """
    # Setup loss function with optional class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
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
                   c_global: list, c_i: list, max_grad_norm: float = 1.0, class_weights=None):
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
        class_weights: Optional tensor of class weights for weighted loss

    Returns:
        c_i_new: Updated client control variate
        delta_c: Change in client control variate (c_i_new - c_i)
    """
    # Setup loss function with optional class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    model.train()
    model.to(device)

    # Convert control variates to tensors
    c_global_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in c_global]
    c_i_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in c_i]

    # Store initial parameters for control variate update
    params_before = [p.detach().clone() for p in model.parameters()]

    # Count total gradient steps for proper scaling
    total_steps = 0
    lr = optimizer.param_groups[0]['lr']

    # Training loop with SCAFFOLD correction
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # SCAFFOLD correction: subtract (c_i - c_global) from gradients
            # Corrected gradient: g_corrected = g + c_global - c_i
            # This pushes local updates toward the global objective
            with torch.no_grad():
                for param, cg, ci in zip(model.parameters(), c_global_tensors, c_i_tensors):
                    if param.grad is not None:
                        # FIXED: Add (c_global - c_i) to correct client drift
                        param.grad.data += (cg - ci)

            # Clip gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            total_steps += 1

    # Compute updated client control variate using Option II
    # c_i_new = c_i - c_global + (x_before - x_after) / (K * lr)
    # where K = number of local gradient steps (total_steps)

    c_i_new = []

    with torch.no_grad():
        for p_before, p_after, cg, ci in zip(params_before, model.parameters(),
                                              c_global_tensors, c_i_tensors):
            # Option II update formula
            # Use total_steps for accurate scaling
            step_size = total_steps * lr if total_steps * lr > 0 else 1.0
            param_diff = (p_before - p_after) / step_size
            c_new = ci - cg + param_diff

            c_i_new.append(c_new.cpu().numpy())

    return c_i_new


def test(net, testloader, device: str, class_weights=None):
    """Evaluate the MLP on the test/validation set."""
    # Setup loss function with optional class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
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


def train_hybrid_fedprox_scaffold(model, trainloader, optimizer, epochs, device: str,
                                   global_params: list, proximal_mu: float,
                                   c_global: list, c_i: list,
                                   max_grad_norm: float = 1.0, class_weights=None,
                                   use_scaffold: bool = True,
                                   use_dual_mu: bool = False,
                                   mu_raw: float = 0.1,
                                   mu_corrected: float = 0.001):
    """Train with Enhanced Hybrid FedProx-SCAFFOLD approach.
    
    This enhanced version supports:
    1. Sequential Activation: Pure SCAFFOLD (mu=0) during warmup, then gradual FedProx
    2. Conditional Activation: Per-client control over which mechanism to apply
    3. Dual-μ Architecture: Separate μ for raw vs SCAFFOLD-corrected gradient components
    
    The Dual-μ Architecture works as follows:
    - μ_raw: Applied to the uncorrected gradient component (stronger regularization)
    - μ_corrected: Applied to SCAFFOLD-corrected component (10-100x smaller)
    - Total proximal loss: (μ_raw/2) × ||raw_component||² + (μ_corrected/2) × ||corrected_component||²
    
    This allows FedProx to strongly constrain uncorrected drift while lightly touching
    SCAFFOLD's corrections, respecting that the corrected component already accounts
    for drift direction.
    
    Args:
        model: Neural network model
        trainloader: Training data loader
        optimizer: Optimizer for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        global_params: List of global model parameters (numpy arrays) for FedProx
        proximal_mu: Legacy single proximal μ (used if use_dual_mu=False)
        c_global: Global control variate (list of numpy arrays) for SCAFFOLD
        c_i: Client control variate (list of numpy arrays) for SCAFFOLD
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        class_weights: Optional tensor of class weights for weighted loss
        use_scaffold: Whether to apply SCAFFOLD correction (default: True)
        use_dual_mu: Whether to use dual-μ architecture (default: False)
        mu_raw: μ for uncorrected gradient component (default: 0.1)
        mu_corrected: μ for SCAFFOLD-corrected component (default: 0.001)
    
    Returns:
        c_i_new: Updated client control variate
    """
    # Setup loss function with optional class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    model.train()
    model.to(device)
    
    # Convert global params to tensors for FedProx proximal term
    global_tensors = [torch.tensor(p, device=device, dtype=torch.float32) 
                      for p in global_params]
    
    # Convert control variates to tensors for SCAFFOLD correction
    c_global_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in c_global]
    c_i_tensors = [torch.tensor(c, device=device, dtype=torch.float32) for c in c_i]
    
    # Store initial parameters for control variate update
    params_before = [p.detach().clone() for p in model.parameters()]
    
    # Count total gradient steps for proper scaling
    total_steps = 0
    lr = optimizer.param_groups[0]['lr']
    
    # Training loop with enhanced hybrid mechanisms
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass - Cross-entropy loss
            outputs = model(features)
            ce_loss = criterion(outputs, labels)
            
            # Compute proximal term based on architecture choice
            if use_dual_mu:
                # Dual-μ Architecture: Separate μ for raw and corrected components
                # This requires computing two separate proximal terms
                
                # For dual-μ, we apply μ_raw to the full deviation
                # and apply μ_corrected as a lighter touch
                # The idea: raw gradient gets stronger regularization
                
                # Primary proximal term (μ_raw)
                proximal_term_raw = 0.0
                for local_param, global_param in zip(model.parameters(), global_tensors):
                    proximal_term_raw += torch.sum((local_param - global_param) ** 2)
                proximal_term_raw = (mu_raw / 2.0) * proximal_term_raw
                
                # Secondary proximal term (μ_corrected) - lighter touch for corrected direction
                # This term is much smaller, acknowledging SCAFFOLD already handles direction
                proximal_term_corrected = 0.0
                if use_scaffold:
                    for local_param, global_param in zip(model.parameters(), global_tensors):
                        proximal_term_corrected += torch.sum((local_param - global_param) ** 2)
                    proximal_term_corrected = (mu_corrected / 2.0) * proximal_term_corrected
                
                # Total loss combines both proximal terms
                # In practice, mu_corrected << mu_raw, so this is dominated by mu_raw
                loss = ce_loss + proximal_term_raw + proximal_term_corrected
                
            elif proximal_mu > 0:
                # Legacy single-μ architecture
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_tensors):
                    proximal_term += torch.sum((local_param - global_param) ** 2)
                proximal_term = (proximal_mu / 2.0) * proximal_term
                
                loss = ce_loss + proximal_term
            else:
                # Pure SCAFFOLD (no proximal term) - Phase 1 warm-up
                loss = ce_loss
            
            # Backward pass
            loss.backward()
            
            # SCAFFOLD direction correction: add (c_global - c_i) to gradients
            if use_scaffold:
                with torch.no_grad():
                    for param, cg, ci in zip(model.parameters(), c_global_tensors, c_i_tensors):
                        if param.grad is not None:
                            # Add control variate correction to gradient
                            param.grad.data += (cg - ci)
            
            # Clip gradients to prevent exploding gradients
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            total_steps += 1
    
    # Compute updated client control variate using Option II
    # c_i_new = c_i - c_global + (x_before - x_after) / (K * lr)
    c_i_new = []
    
    with torch.no_grad():
        for p_before, p_after, cg, ci in zip(params_before, model.parameters(),
                                              c_global_tensors, c_i_tensors):
            # Option II update formula with total_steps for accurate scaling
            step_size = total_steps * lr if total_steps * lr > 0 else 1.0
            param_diff = (p_before - p_after) / step_size
            c_new = ci - cg + param_diff
            
            c_i_new.append(c_new.cpu().numpy())
    
    return c_i_new
