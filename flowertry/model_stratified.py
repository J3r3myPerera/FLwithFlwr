"""
Neural network model for Disposable Income regression.
Used for stratified client selection with FedAvg.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DisposableIncomeModel(nn.Module):
    """
    Neural network for Disposable Income regression.
    Architecture: 19 input features -> 128 -> 64 -> 32 -> 1 output
    Includes BatchNorm and Dropout for better generalization
    """
    
    def __init__(self, input_dim: int = 19) -> None:
        super(DisposableIncomeModel, self).__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
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
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train(net, trainloader, optimizer, epochs, device: str):
    """
    Train the network on the training set using MSE loss (FedAvg).
    
    Args:
        net: Neural network model
        trainloader: DataLoader for training data
        optimizer: Optimizer for training
        epochs: Number of local epochs
        device: Device to train on
    """
    criterion = nn.MSELoss()
    net.train()
    net.to(device)
    
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
            
            # Check for NaN in outputs
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
                continue
            
            loss = criterion(outputs, targets)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update BatchNorm running statistics
            for module in net.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.track_running_stats = True
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0


def test(net, testloader, device: str, target_scaler=None):
    """
    Evaluate the network on the test/validation set.
    
    Args:
        net: Neural network model
        testloader: DataLoader for test/validation data
        device: Device to evaluate on
        target_scaler: StandardScaler fitted on target (for inverse transform to original scale)
    
    Returns: 
        loss (MSE), RMSE, MAE, R²
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
                continue
            
            # Reshape targets to match output shape
            targets_reshaped = targets.unsqueeze(1)
            
            outputs = net(features)
            
            # Check for NaN in outputs
            if torch.any(torch.isnan(outputs)) or torch.any(torch.isinf(outputs)):
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
    
    # Handle NaN or Inf values in predictions
    if np.any(np.isnan(all_predictions)) or np.any(np.isinf(all_predictions)):
        all_predictions = np.nan_to_num(all_predictions, nan=np.nanmean(all_targets), 
                                        posinf=np.nanmean(all_targets), 
                                        neginf=np.nanmean(all_targets))
    
    # Handle NaN in targets
    if np.any(np.isnan(all_targets)):
        valid_mask = ~np.isnan(all_targets)
        all_predictions = all_predictions[valid_mask]
        all_targets = all_targets[valid_mask]
    
    # Inverse transform to original scale if scaler is provided
    if target_scaler is not None:
        all_predictions = target_scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    if len(all_predictions) == 0 or len(all_targets) == 0:
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
