import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DisposableIncomeModel(nn.Module):
    """
    Improved neural network for Disposable Income regression.
    Architecture: 19 input features -> 128 -> 64 -> 32 -> 1 output
    Includes BatchNorm and Dropout for better generalization
    """
    
    def __init__(self, input_dim: int = 19) -> None:
        super(DisposableIncomeModel, self).__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),  # More stable BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third hidden layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(32, 1)
        )
        
        # Initialize weights properly to avoid NaN
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for ReLU layers
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                # Initialize BatchNorm parameters
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure BatchNorm uses running statistics in eval mode
        # This prevents issues with small batch sizes
        return self.layers(x)


def train(net, trainloader, optimizer, epochs, device: str, mu: float = 0.0, global_params=None, layer_mus=None):
    """
    Train the network on the training set using MSE loss.
    
    Args:
        net: Neural network model
        trainloader: DataLoader for training data
        optimizer: Optimizer for training
        epochs: Number of local epochs
        device: Device to train on
        mu: FedProx proximal term coefficient (0.0 for FedAvg) - used if layer_mus is None
        global_params: Global model parameters for FedProx
        layer_mus: Dict mapping layer names to mu values for multi-layer FedProx.
                   If None, uses single mu for all layers (base FedProx).
                   Layer names: 'input', 'hidden1', 'hidden2', 'hidden3', 'output'
    """
    criterion = nn.MSELoss()
    net.train()
    net.to(device)
    
    # Store global parameters for FedProx if provided
    if global_params is not None and mu > 0:
        # Convert to float32 for MPS compatibility (MPS doesn't support float64)
        global_params_tensor = [torch.tensor(p, dtype=torch.float32, device=device) for p in global_params]
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for features, targets in trainloader:
            features, targets = features.to(device), targets.to(device)
            
            # Check for NaN in input features
            if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
                continue  # Skip this batch
            
            # Reshape targets to match output shape [batch_size, 1]
            targets = targets.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = net(features)
            
            # Check for NaN in outputs before computing loss
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                # Skip this batch if outputs are invalid
                continue
            
            loss = criterion(outputs, targets)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # Skip this batch
            
            # Add FedProx proximal term if mu > 0 or layer_mus is provided
            use_fedprox = (mu > 0 or (layer_mus is not None and any(v > 0 for v in layer_mus.values())))
            
            if use_fedprox and global_params is not None:
                try:
                    proximal_term = 0.0
                    current_params = list(net.parameters())
                    param_names = [name for name, _ in net.named_parameters()]
                    valid_proximal = True
                    
                    # Map parameter names to layer names for multi-layer mu
                    # Model structure: layers.0 (input), layers.4 (hidden1), layers.8 (hidden2), layers.12 (output)
                    def get_layer_name(param_name):
                        """Map parameter name to layer name"""
                        if 'layers.0' in param_name or 'layers.1' in param_name:
                            return 'input'
                        elif 'layers.4' in param_name or 'layers.5' in param_name:
                            return 'hidden1'
                        elif 'layers.8' in param_name or 'layers.9' in param_name:
                            return 'hidden2'
                        elif 'layers.12' in param_name:
                            return 'output'
                        else:
                            return 'output'  # Default fallback
                    
                    for idx, (curr_param, global_param) in enumerate(zip(current_params, global_params_tensor)):
                        diff = curr_param - global_param
                        # Check for NaN in difference
                        if torch.any(torch.isnan(diff)) or torch.any(torch.isinf(diff)):
                            valid_proximal = False
                            break
                        
                        norm_sq = torch.norm(diff) ** 2
                        if torch.isnan(norm_sq) or torch.isinf(norm_sq):
                            valid_proximal = False
                            break
                        
                        # Determine mu value for this parameter
                        if layer_mus is not None:
                            # Use layer-specific mu based on parameter name
                            param_name = param_names[idx]
                            layer_name = get_layer_name(param_name)
                            param_mu = layer_mus.get(layer_name, mu)  # Fallback to base mu if layer not found
                        else:
                            # Use base mu for all parameters
                            param_mu = mu
                        
                        # Add weighted proximal term
                        proximal_term += (param_mu / 2.0) * norm_sq
                    
                    # Check if proximal term is valid before adding to loss
                    if valid_proximal and not (torch.isnan(proximal_term) or torch.isinf(proximal_term)):
                        loss += proximal_term
                except Exception as e:
                    # If proximal term calculation fails, continue without it
                    pass
            
            # Check loss again after adding proximal term
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # Skip this batch
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update BatchNorm running statistics manually if needed
            # This helps with stability during federated learning
            for module in net.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.track_running_stats = True
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        # Optionally print training progress
        # print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


def test(net, testloader, device: str, target_scaler=None):
    """
    Evaluate the network on the test/validation set.
    
    Args:
        net: Neural network model
        testloader: DataLoader for test/validation data
        device: Device to evaluate on
        target_scaler: StandardScaler fitted on target (for inverse transform to original scale)
    
    Returns: loss (MSE), RMSE, MAE, R²
    Note: If target_scaler is provided, metrics are computed in original scale
    """
    criterion = nn.MSELoss()
    net.eval()
    net.to(device)
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for features, targets in testloader:
            features, targets = features.to(device), targets.to(device)
            
            # Check for NaN in input features
            if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
                print(f"Warning: Input features contain NaN/Inf. Skipping batch.")
                continue
            
            # Reshape targets to match output shape
            targets_reshaped = targets.unsqueeze(1)
            
            outputs = net(features)
            
            # Check for NaN in outputs before computing loss
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                print(f"Warning: Model outputs contain NaN/Inf. Using target mean as prediction.")
                outputs = torch.full_like(outputs, targets_reshaped.mean().item())
            
            loss = criterion(outputs, targets_reshaped)
            
            # Skip if loss is NaN
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
            
            # Collect predictions and targets for metrics
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Check for NaN or Inf values in predictions
    if np.any(np.isnan(all_predictions)) or np.any(np.isinf(all_predictions)):
        # If predictions contain NaN/Inf, replace with mean of targets as fallback
        print(f"Warning: Predictions contain NaN/Inf values. Using target mean as fallback.")
        all_predictions = np.nan_to_num(all_predictions, nan=np.nanmean(all_targets), 
                                        posinf=np.nanmean(all_targets), 
                                        neginf=np.nanmean(all_targets))
    
    # Check for NaN in targets (shouldn't happen, but just in case)
    if np.any(np.isnan(all_targets)):
        print(f"Warning: Targets contain NaN values. Filtering them out.")
        valid_mask = ~np.isnan(all_targets)
        all_predictions = all_predictions[valid_mask]
        all_targets = all_targets[valid_mask]
    
    # Inverse transform to original scale if scaler is provided
    if target_scaler is not None:
        all_predictions = target_scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    if len(all_predictions) == 0 or len(all_targets) == 0:
        # Return high error values if no valid data
        mse = 1e10
        rmse = 1e5
        mae = 1e5
        r2 = -1e10
    else:
        # Compute MSE in original scale if scaler was used
        if target_scaler is not None:
            mse = mean_squared_error(all_targets, all_predictions)
        else:
            mse = total_loss / n_batches if n_batches > 0 else 0.0
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
    
    return mse, rmse, mae, r2
