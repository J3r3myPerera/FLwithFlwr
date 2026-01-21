"""
Neural Network Models for Federated Disposable Income Regression.

Improved FL-optimized architecture:
- Input: 25 features (12 base + 6 engineered + 7 one-hot encoded)
  * Base numerical: Age, Dependents, 10 expense categories
  * Engineered: Total_Expenses, Expense_to_Income_Ratio, Essential_Expenses,
                Discretionary_Expenses, Age_squared, Income_log
  * Categorical one-hot: Occupation (4 categories), City_Tier (3 categories)
- LayerNorm for feature normalization (FL-safe, no global stats)
- Simpler 2-layer architecture: 128 -> 64 neurons
- GELU activation (smoother than ReLU, better for client drift)
- Dropout (0.15) for regularization (handles heterogeneity)
- Output: 1 neuron (continuous regression output)

Key FL Improvements:
1. LayerNorm instead of BatchNorm (no reliance on global statistics)
2. GELU activation (smoother gradient flow, more robust to client drift)
3. Dropout for regularization (better generalization across heterogeneous clients)
4. Wider network with proper initialization
5. AdamW optimizer with cosine annealing (adaptive learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from collections import OrderedDict


class DisposableIncomeNet(nn.Module):
    """
    FL-Optimized Feedforward Neural Network for Disposable Income Regression.
    
    Architecture (Enhanced for Hybrid FL):
        Linear(25, 160) -> LayerNorm -> GELU -> Dropout ->
        Linear(160, 96) -> LayerNorm -> GELU -> Dropout ->
        Linear(96, 48) -> LayerNorm -> GELU -> Dropout ->
        Linear(48, 1)
    
    Features:
    - LayerNorm: Normalizes within each sample (FL-safe, no global preprocessing)
    - GELU: Smooth activation function (more robust to client drift than ReLU)
    - Dropout: Regularization for heterogeneous data
    - Wider & Deeper: 3-layer architecture (160->96->48) for better capacity
    - Residual connections for gradient flow (optional)
    """
    
    def __init__(
        self, 
        input_dim: int = 25,  # Updated: 12 base + 6 engineered + 7 one-hot
        hidden_dim1: int = 96,   # Reduced from 160 to prevent overfitting
        hidden_dim2: int = 48,   # Reduced from 96 to prevent overfitting
        hidden_dim3: int = 24,   # Reduced from 48 to prevent overfitting
        dropout: float = 0.40,   # Significantly increased (0.18->0.40) for strong regularization
        use_layer_norm: bool = True,
        use_residual: bool = False  # Optional residual connections
    ):
        """
        Initialize the FL-optimized regression network.
        
        Args:
            input_dim: Number of input features (default 25: 12 base + 6 engineered + 7 one-hot)
            hidden_dim1: First hidden layer size (default 96, reduced from 160)
            hidden_dim2: Second hidden layer size (default 48, reduced from 96)
            hidden_dim3: Third hidden layer size (default 24, reduced from 48)
            dropout: Dropout rate for regularization (default 0.40, increased from 0.18)
            use_layer_norm: Whether to use LayerNorm (default True, recommended for FL)
            use_residual: Whether to use residual connections (default False)
        """
        super(DisposableIncomeNet, self).__init__()
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        if use_layer_norm:
            self.ln2 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3 (NEW for better capacity)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        if use_layer_norm:
            self.ln3 = nn.LayerNorm(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim3, 1)
        
        # Initialize weights for better convergence with GELU
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization for GELU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization works well with GELU
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU activation and optional LayerNorm.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Layer 1: Linear -> LayerNorm -> GELU -> Dropout
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.gelu(x)  # GELU: smoother than ReLU, better for FL
        x = self.dropout1(x)
        
        # Layer 2: Linear -> LayerNorm -> GELU -> Dropout
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        # Layer 3: Linear -> LayerNorm -> GELU -> Dropout (NEW)
        x = self.fc3(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        # Output layer (no activation for regression)
        x = self.fc4(x)
        return x


# Alias for compatibility
Net = DisposableIncomeNet


class LegacyDisposableIncomeNet(nn.Module):
    """
    Legacy model architecture (for comparison or backwards compatibility).
    
    Simple architecture without FL optimizations:
        Linear(19, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)
    """
    
    def __init__(self, input_dim: int = 19, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super(LegacyDisposableIncomeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List) -> None:
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def create_model(input_dim: int = 19, use_fl_optimizations: bool = True) -> nn.Module:
    """
    Factory function to create a model instance.
    
    Args:
        input_dim: Number of input features
        use_fl_optimizations: If True, use FL-optimized model with LayerNorm, GELU, Dropout.
                             If False, use legacy model with ReLU.
    
    Returns:
        Model instance
    """
    if use_fl_optimizations:
        return DisposableIncomeNet(input_dim=input_dim)
    else:
        return LegacyDisposableIncomeNet(input_dim=input_dim)


if __name__ == "__main__":
    print("Testing FL-Optimized Disposable Income Regression Model")
    print("=" * 70)
    
    # Test FL-optimized model
    print("\n1. FL-Optimized Model (with LayerNorm, GELU, Dropout)")
    print("-" * 70)
    model = DisposableIncomeNet(input_dim=19)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: 19 -> 160 -> 96 -> 48 -> 1")
    
    # Test forward pass
    x = torch.randn(8, 19)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test with training mode vs eval mode (dropout behavior)
    model.train()
    y_train = model(x)
    model.eval()
    y_eval = model(x)
    print(f"Train mode output (with dropout): {y_train[:3].flatten()}")
    print(f"Eval mode output (no dropout): {y_eval[:3].flatten()}")
    print(f"Outputs differ (dropout active in train): {not torch.allclose(y_train, y_eval)}")
    
    # Test parameter extraction
    params = get_parameters(model)
    print(f"\nNumber of parameter tensors: {len(params)}")
    
    # Test parameter setting
    model2 = DisposableIncomeNet(input_dim=19)
    set_parameters(model2, params)
    print("Parameter transfer successful!")
    
    # Compare with legacy model
    print("\n2. Legacy Model (for comparison)")
    print("-" * 70)
    legacy_model = LegacyDisposableIncomeNet(input_dim=19)
    print(f"Model: {legacy_model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in legacy_model.parameters()):,}")
    print(f"Architecture: 19 -> 64 -> 32 -> 1")
    
    print("\n" + "=" * 70)
    print("Key FL Improvements:")
    print("  ✓ LayerNorm: No reliance on global preprocessing")
    print("  ✓ GELU: Smoother activation, more robust to client drift")
    print("  ✓ Dropout: Better regularization for heterogeneous data")
    print("  ✓ Simpler 2-layer: Faster training, less overfitting (128->64 vs 64->32)")
    print("=" * 70)
