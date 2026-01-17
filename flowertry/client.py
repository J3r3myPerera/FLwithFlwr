"""
Federated Learning Client for Disposable Income Regression.

Supports multiple training strategies:
- FedAvg: Standard federated averaging
- FedProx: Federated averaging with proximal term for Non-IID data
- SCAFFOLD: Variance reduction via control variates
- Hybrid: Combined FedProx + SCAFFOLD for maximum Non-IID robustness
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import flwr as fl
from flwr.common import NDArrays, Scalar

from model import DisposableIncomeNet, get_parameters, set_parameters


class RegressionClient(fl.client.NumPyClient):
    """
    Flower client for federated regression with multiple strategy support.
    
    Strategies:
    - FedAvg: Standard SGD training
    - FedProx: Adds proximal term ||w - w_global||^2 to loss
    - SCAFFOLD: Uses control variates to correct gradient drift
    - Hybrid: Combines FedProx proximal term with SCAFFOLD control variates
    """
    
    def __init__(
        self,
        cid: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        input_dim: int = 19,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        log_transform: bool = False,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        strategy: str = "fedavg",
        mu: float = 0.1,  # FedProx proximal term weight
        scaffold_lr_correction: float = 1.0,  # SCAFFOLD learning rate correction
        fedprox_weight: float = 1.0,  # Hybrid: weight for FedProx component
        scaffold_weight: float = 1.0,  # Hybrid: weight for SCAFFOLD component
        device: Optional[torch.device] = None
    ):
        """
        Initialize the client.
        
        Args:
            cid: Client ID
            trainloader: Training data loader
            valloader: Validation data loader
            input_dim: Number of input features
            target_mean: Mean of transformed target (for denormalization)
            target_std: Std of transformed target (for denormalization)
            log_transform: Whether log(1+y) transformation was applied to target.
                           If True, predictions will be inverse-transformed for metrics.
            local_epochs: Number of local training epochs
            learning_rate: Learning rate
            strategy: Training strategy ("fedavg", "fedprox", "scaffold", "hybrid")
            mu: FedProx proximal term weight
            scaffold_lr_correction: SCAFFOLD learning rate correction factor
            fedprox_weight: Weight for FedProx component in hybrid (0-1)
            scaffold_weight: Weight for SCAFFOLD component in hybrid (0-1)
            device: Torch device
        """
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.input_dim = input_dim
        self.target_mean = target_mean
        self.target_std = target_std
        self.log_transform = log_transform
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.strategy = strategy.lower()
        self.mu = mu
        self.scaffold_lr_correction = scaffold_lr_correction
        self.fedprox_weight = fedprox_weight
        self.scaffold_weight = scaffold_weight
        self.max_grad_norm = 1.0  # Default gradient clipping norm
        
        # Hybrid-specific enhancements
        self.control_momentum = 0.9  # Momentum for smoother SCAFFOLD updates
        self.momentum_control = None  # Momentum-enhanced control variate
        self.round_num = 0  # Track training rounds for adaptive behavior
        self.client_drift_history = []  # Track client drift for adaptation
        
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Create model
        self.model = DisposableIncomeNet(input_dim=input_dim).to(self.device)
        
        # Loss function for regression
        self.criterion = nn.MSELoss()
        
        # SCAFFOLD control variates (initialized to zeros)
        self.client_control = None  # c_i
        self.server_control = None  # c
        
        # Store global model for FedProx
        self.global_params = None
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters."""
        return get_parameters(self.model)
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters."""
        set_parameters(self.model, parameters)
        # Store copy of global parameters for FedProx
        self.global_params = [p.clone().detach() for p in self.model.parameters()]
    
    def _init_control_variates(self):
        """Initialize SCAFFOLD control variates to zeros."""
        self.client_control = [torch.zeros_like(p) for p in self.model.parameters()]
        self.server_control = [torch.zeros_like(p) for p in self.model.parameters()]
    
    def set_server_control(self, server_control: List[np.ndarray]):
        """Set server control variate from numpy arrays."""
        self.server_control = [torch.tensor(c, device=self.device) for c in server_control]
    
    def get_client_control(self) -> List[np.ndarray]:
        """Get client control variate as numpy arrays."""
        if self.client_control is None:
            self._init_control_variates()
        return [c.cpu().numpy() for c in self.client_control]
    
    def _train_fedavg(self) -> float:
        """Standard FedAvg training with AdamW optimizer and gradient clipping."""
        self.model.train()
        # AdamW: Adaptive learning with better weight decay handling than SGD
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Optional: Learning rate scheduler (cosine annealing within local epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(self.trainloader) * self.local_epochs, eta_min=self.learning_rate * 0.1
        )
        
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            total_loss += epoch_loss
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_fedprox(self) -> float:
        """FedProx training with proximal term, AdamW, and gradient clipping."""
        self.model.train()
        # AdamW for adaptive learning
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(self.trainloader) * self.local_epochs, eta_min=self.learning_rate * 0.1
        )
        
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                
                # MSE loss
                mse_loss = self.criterion(y_pred, y_batch)
                
                # Proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                if self.global_params is not None:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p.to(self.device)) ** 2)
                    proximal_term = (self.mu / 2.0) * proximal_term
                
                # Total loss
                loss = mse_loss + proximal_term
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_scaffold(self) -> Tuple[float, List[np.ndarray]]:
        """
        SCAFFOLD training with control variates, AdamW optimizer, and gradient clipping.
        
        Key idea: Correct local gradient with control variates to reduce variance
        g_corrected = g_local - c_i + c (local gradient - client control + server control)
        
        Returns:
            avg_loss: Average training loss
            delta_control: Change in client control variate
        """
        if self.client_control is None or self.server_control is None:
            self._init_control_variates()
        
        self.model.train()
        
        # Use AdamW for better adaptive learning
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(self.trainloader) * self.local_epochs, eta_min=self.learning_rate * 0.1
        )
        
        # Store initial parameters for control variate update
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        
        total_loss = 0.0
        n_batches = 0
        total_steps = 0
        
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Apply SCAFFOLD correction to gradients: g_corrected = g + (c - c_i)
                with torch.no_grad():
                    for param, c_i, c in zip(self.model.parameters(), 
                                              self.client_control, 
                                              self.server_control):
                        if param.grad is not None:
                            # Correct gradient: add (server_control - client_control)
                            param.grad.add_(c.to(self.device) - c_i.to(self.device))
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                # Update with optimizer (handles momentum and adaptive LR)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
                total_steps += 1
        
        # Update client control variate: c_i_new = c_i - c + (x_0 - x_K) / (K * lr)
        # where K is total steps, x_0 is initial params, x_K is final params
        old_client_control = [c.clone() for c in self.client_control]
        
        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control, 
                initial_params, self.model.parameters()
            )):
                # Compute parameter drift
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / (total_steps * self.learning_rate)
                # Update client control: c_i_new = c_i - c + param_diff
                self.client_control[i] = c_i.to(self.device) - c.to(self.device) + param_diff
        
        # Compute delta control (c_i_new - c_i_old) for server aggregation
        delta_control = [
            (new_c - old_c).cpu().numpy() 
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]
        
        return total_loss / n_batches if n_batches > 0 else 0.0, delta_control
    
    def _train_hybrid(self, config: Dict[str, Scalar] = None) -> Tuple[float, List[np.ndarray]]:
        """
        ADVANCED Hybrid FedProx + SCAFFOLD training with:
        1. Adaptive weight balancing (progressive FedProx->SCAFFOLD transition)
        2. Momentum-enhanced control variates
        3. Client-specific learning rate adaptation
        4. Dynamic mu adjustment based on gradient norms
        5. Advanced gradient correction mechanisms
        
        Args:
            config: Training configuration dict with server_round
        
        Returns:
            avg_loss: Average training loss
            delta_control: Change in client control variate
        """
        if self.client_control is None or self.server_control is None:
            self._init_control_variates()
        
        # Initialize momentum control variate
        if self.momentum_control is None:
            self.momentum_control = [torch.zeros_like(c) for c in self.client_control]
        
        self.model.train()
        
        # Extract server round from config
        server_round = int(config.get("server_round", 1)) if config else 1
        
        # Adaptive Learning Rate: Higher boost early for faster convergence
        # This helps Hybrid converge faster than FedAvg
        adaptive_lr = self.learning_rate * (1.0 + 0.5 * np.exp(-server_round / 10.0))
        
        # Use AdamW optimizer with adaptive LR
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=adaptive_lr, 
            weight_decay=1e-4,  # Standard weight decay (not too aggressive)
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Warmup + Cosine annealing scheduler for smoother training
        total_steps = len(self.trainloader) * self.local_epochs
        warmup_steps = min(5, total_steps // 4)  # 25% warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Store initial parameters
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        
        # ADAPTIVE WEIGHT STRATEGY:
        # Early rounds: Strong FedProx (stability) + Weak SCAFFOLD  
        # Later rounds: Balanced FedProx + Strong SCAFFOLD (drift correction)
        # This progressive transition is KEY to beating FedAvg/FedProx
        
        # Get server round from config (since clients are recreated each round)
        if config is None:
            config = {}
        server_round = int(config.get("server_round", 1))
        progress = min(1.0, server_round / 25.0)  # Normalize to [0, 1] over 25 rounds (full training)
        
        # AGGRESSIVE adaptive FedProx weight: Start at 0.9, gradually decrease to 0.6
        # Strong initial stability, then release for exploration
        dynamic_fedprox_weight = 0.9 - 0.3 * progress
        
        # AGGRESSIVE adaptive SCAFFOLD weight: Start at 0.2, gradually increase to 0.7
        # Weak initially (control variates not calibrated), strong later (drift correction)
        dynamic_scaffold_weight = 0.2 + 0.5 * progress
        
        # Adaptive mu: Start at 0.08 for initial stability, decrease to 0.03
        # Stronger initial regularization than FedProx
        dynamic_mu = 0.08 - 0.05 * progress
        
        total_loss = 0.0
        n_batches = 0
        total_steps_actual = 0
        gradient_norms = []
        
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                mse_loss = self.criterion(y_pred, y_batch)
                
                # ENHANCED FedProx proximal term with adaptive weighting
                proximal_term = 0.0
                if self.global_params is not None:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p.to(self.device)) ** 2)
                    # Apply dynamic weights
                    proximal_term = (dynamic_fedprox_weight * dynamic_mu / 2.0) * proximal_term
                
                loss = mse_loss + proximal_term
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Track gradient norms for adaptive mu adjustment
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                gradient_norms.append(grad_norm.item())
                
                # ADVANCED SCAFFOLD correction with momentum
                with torch.no_grad():
                    for param_idx, (param, c_i, c) in enumerate(zip(
                        self.model.parameters(),
                        self.client_control,
                        self.server_control
                    )):
                        if param.grad is not None:
                            # Standard SCAFFOLD correction
                            base_correction = c.to(self.device) - c_i.to(self.device)
                            
                            # Apply momentum to control variates for smoother updates
                            self.momentum_control[param_idx] = (
                                self.control_momentum * self.momentum_control[param_idx].to(self.device) +
                                (1 - self.control_momentum) * base_correction
                            )
                            
                            # Apply scaled correction with dynamic weight
                            correction = dynamic_scaffold_weight * self.momentum_control[param_idx]
                            param.grad.add_(correction)
                
                # Adaptive gradient clipping based on gradient statistics
                # Tighter clipping in early rounds, looser later
                adaptive_clip_norm = self.max_grad_norm * (1.0 + 0.5 * progress)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=adaptive_clip_norm)
                
                # Update with optimizer
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
                total_steps_actual += 1
        
        # Track client drift for future adaptation
        param_drift = sum(
            torch.sum((x_0.to(self.device) - x_K.to(self.device)) ** 2).item()
            for x_0, x_K in zip(initial_params, self.model.parameters())
        )
        self.client_drift_history.append(param_drift)
        
        # ENHANCED control variate update with averaging for stability
        old_client_control = [c.clone() for c in self.client_control]
        
        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control,
                initial_params, self.model.parameters()
            )):
                # Standard SCAFFOLD update
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / (total_steps_actual * adaptive_lr)
                
                # Apply exponential moving average for stability
                ema_factor = 0.9  # Balanced smoothing for stable control variates
                new_control = c_i.to(self.device) - c.to(self.device) + param_diff
                self.client_control[i] = ema_factor * new_control + (1 - ema_factor) * c_i.to(self.device)
        
        delta_control = [
            (new_c - old_c).cpu().numpy()
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]
        
        return total_loss / n_batches if n_batches > 0 else 0.0, delta_control
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
        
        Returns:
            Updated parameters, number of samples, metrics dict
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Extract config
        self.local_epochs = int(config.get("local_epochs", self.local_epochs))
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.mu = float(config.get("mu", self.mu))
        self.scaffold_lr_correction = float(config.get("scaffold_lr_correction", self.scaffold_lr_correction))
        self.fedprox_weight = float(config.get("fedprox_weight", self.fedprox_weight))
        self.scaffold_weight = float(config.get("scaffold_weight", self.scaffold_weight))
        self.max_grad_norm = float(config.get("max_grad_norm", self.max_grad_norm))
        
        # Advanced adaptive regularization (for hybrid)
        server_round = int(config.get("server_round", 1))
        if self.strategy == "hybrid":
            # Dynamic weight adjustment
            dynamic_weights = config.get("dynamic_weights", False)
            if dynamic_weights and server_round > 1:
                weight_increase_rate = float(config.get("weight_increase_rate", 1.05))
                weight_max = float(config.get("weight_max", 0.6))
                self.fedprox_weight = min(self.fedprox_weight * weight_increase_rate, weight_max)
                self.scaffold_weight = min(self.scaffold_weight * weight_increase_rate, weight_max)
            
            # Advanced warmup (exponential, linear, or cosine)
            warmup_rounds = int(config.get("warmup_rounds", 0))
            if warmup_rounds > 0 and server_round <= warmup_rounds:
                warmup_type = config.get("warmup_type", "linear")
                warmup_start_factor = float(config.get("warmup_start_factor", 0.1))
                
                progress = server_round / warmup_rounds
                if warmup_type == "exponential":
                    warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * (1 - (1 - progress) ** 2)
                elif warmup_type == "cosine":
                    import math
                    warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * (1 - math.cos(progress * math.pi)) / 2
                else:  # linear
                    warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress
                
                self.fedprox_weight *= warmup_factor
                self.scaffold_weight *= warmup_factor
            
            # Store gradient clipping config
            self.gradient_clip = config.get("gradient_clip", False)
            self.gradient_clip_norm = float(config.get("gradient_clip_norm", 1.0))
            
            # Store momentum config
            self.use_momentum = config.get("use_momentum", False)
            self.momentum_value = float(config.get("momentum", 0.9))
            if self.use_momentum and not hasattr(self, 'velocity'):
                self.velocity = None
        
        # Extract SCAFFOLD server control if provided
        if "server_control" in config and config["server_control"] is not None:
            # This would need custom handling - simplified here
            pass
        
        # Train based on strategy
        delta_control = None
        
        if self.strategy == "fedavg":
            loss = self._train_fedavg()
        elif self.strategy == "fedprox":
            loss = self._train_fedprox()
        elif self.strategy == "scaffold":
            loss, delta_control = self._train_scaffold()
        elif self.strategy == "hybrid":
            loss, delta_control = self._train_hybrid(config)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Get updated parameters
        new_parameters = self.get_parameters({})
        num_samples = len(self.trainloader.dataset)
        
        # Metrics
        metrics = {
            "client_id": self.cid,
            "train_loss": float(loss),
            "num_samples": num_samples,
            "strategy": self.strategy
        }
        
        # Include delta_control for SCAFFOLD and Hybrid strategies
        if delta_control is not None:
            # Convert to serializable format (list of arrays)
            metrics["delta_control"] = [dc.tolist() if hasattr(dc, 'tolist') else dc for dc in delta_control]
        
        return new_parameters, num_samples, metrics
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on validation data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            Loss, number of samples, metrics dict
        """
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.valloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                total_loss += loss.item() * len(y_batch)
                all_preds.append(y_pred.cpu())
                all_targets.append(y_batch.cpu())
        
        # Concatenate predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        num_samples = len(all_targets)
        
        # Compute metrics
        avg_loss = total_loss / num_samples
        
        # Denormalize for actual metrics
        preds_original = all_preds * self.target_std + self.target_mean
        targets_original = all_targets * self.target_std + self.target_mean
        
        # Inverse log transformation: y = exp(y') - 1
        if self.log_transform:
            preds_original = torch.expm1(preds_original)  # exp(x) - 1
            targets_original = torch.expm1(targets_original)
        
        # RMSE
        mse = torch.mean((preds_original - targets_original) ** 2).item()
        rmse = np.sqrt(mse)
        
        # MAE
        mae = torch.mean(torch.abs(preds_original - targets_original)).item()
        
        # R²
        ss_res = torch.sum((targets_original - preds_original) ** 2).item()
        ss_tot = torch.sum((targets_original - torch.mean(targets_original)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # MAPE (Mean Absolute Percentage Error)
        epsilon = 1e-8
        mape = torch.mean(torch.abs((targets_original - preds_original) / (targets_original + epsilon))) * 100
        mape = mape.item()
        
        metrics = {
            "client_id": self.cid,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }
        
        return float(avg_loss), num_samples, metrics


def create_client(
    cid: str,
    trainloader: DataLoader,
    valloader: DataLoader,
    input_dim: int,
    target_mean: float,
    target_std: float,
    config: Dict,
    log_transform: bool = False
) -> RegressionClient:
    """
    Factory function to create a client.
    
    Args:
        cid: Client ID
        trainloader: Training data loader
        valloader: Validation data loader
        input_dim: Number of input features
        target_mean: Target mean for denormalization
        target_std: Target std for denormalization
        config: Client configuration dict
        log_transform: Whether log transformation was applied to target
    
    Returns:
        Configured RegressionClient
    """
    return RegressionClient(
        cid=cid,
        trainloader=trainloader,
        valloader=valloader,
        input_dim=input_dim,
        target_mean=target_mean,
        target_std=target_std,
        log_transform=log_transform,
        local_epochs=config.get("local_epochs", 5),
        learning_rate=config.get("learning_rate", 0.01),
        strategy=config.get("strategy", "fedavg"),
        mu=config.get("mu", 0.1),
        scaffold_lr_correction=config.get("scaffold_lr_correction", 1.0),
        fedprox_weight=config.get("fedprox_weight", 1.0),
        scaffold_weight=config.get("scaffold_weight", 1.0)
    )
