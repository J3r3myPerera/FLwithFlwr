"""
Federated Learning Server for Disposable Income Regression.

Implements server-side aggregation for:
- FedAvg: Standard weighted averaging
- FedProx: Same as FedAvg (proximal term is client-side)
- SCAFFOLD: Averaging with control variate updates
- Hybrid: Combined approach
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Callable, Union
from collections import OrderedDict
import flwr as fl
from flwr.common import (
    Parameters, FitRes, EvaluateRes, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
    FitIns, EvaluateIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, FedAvg

from model import DisposableIncomeNet, get_parameters, set_parameters


def compute_regression_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    target_std: float,
    target_mean: float,
    device: torch.device,
    log_transform: bool = False
) -> Dict[str, float]:
    """
    Compute regression metrics on a dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        target_std: Target standard deviation for denormalization
        target_mean: Target mean for denormalization
        device: Torch device
        log_transform: Whether log(1+y) transformation was applied.
                       If True, predictions will be inverse-transformed.
    
    Returns:
        Dictionary with loss, rmse, mae, r2, mape, accuracy metrics
    """
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    num_samples = len(all_targets)
    
    # Denormalize
    preds_original = all_preds * target_std + target_mean
    targets_original = all_targets * target_std + target_mean
    
    # Inverse log transformation: y = exp(y') - 1
    if log_transform:
        preds_original = torch.expm1(preds_original)  # exp(x) - 1
        targets_original = torch.expm1(targets_original)
    
    # Metrics
    avg_loss = total_loss / num_samples
    mse = torch.mean((preds_original - targets_original) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(preds_original - targets_original)).item()
    
    ss_res = torch.sum((targets_original - preds_original) ** 2).item()
    ss_tot = torch.sum((targets_original - torch.mean(targets_original)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-8
    mape = torch.mean(torch.abs((targets_original - preds_original) / (targets_original + epsilon))) * 100
    mape = mape.item()
    
    # Threshold Accuracy: percentage of predictions within X% of actual
    percentage_errors = torch.abs((targets_original - preds_original) / (targets_original + epsilon)) * 100
    accuracy_10 = (percentage_errors <= 10).float().mean().item() * 100  # Within 10%
    accuracy_20 = (percentage_errors <= 20).float().mean().item() * 100  # Within 20%
    
    return {
        "loss": avg_loss,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "accuracy_10": accuracy_10,
        "accuracy_20": accuracy_20
    }


class FedAvgStrategy(FedAvg):
    """
    FedAvg Strategy with custom evaluation for regression.
    
    This is the baseline federated averaging strategy.
    """
    
    def __init__(
        self,
        testloader: DataLoader,
        input_dim: int,
        target_mean: float,
        target_std: float,
        log_transform: bool = False,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        
        self.testloader = testloader
        self.input_dim = input_dim
        self.target_mean = target_mean
        self.target_std = target_std
        self.log_transform = log_transform
        
        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Create model for evaluation
        self.model = DisposableIncomeNet(input_dim=input_dim).to(self.device)
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test set."""
        # Set parameters
        params = parameters_to_ndarrays(parameters)
        set_parameters(self.model, params)
        
        # Compute metrics
        metrics = compute_regression_metrics(
            self.model, self.testloader,
            self.target_std, self.target_mean, self.device,
            self.log_transform
        )
        
        return metrics["loss"], {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "accuracy_10": metrics["accuracy_10"],
            "accuracy_20": metrics["accuracy_20"]
        }


class FedProxStrategy(FedAvgStrategy):
    """
    FedProx Strategy.
    
    Server-side is identical to FedAvg - the proximal term is applied client-side.
    The key difference is passing the mu parameter to clients.
    """
    
    def __init__(
        self,
        mu: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training with mu parameter."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add mu to config
        config["mu"] = self.mu
        
        fit_ins = FitIns(parameters, config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, fit_ins) for client in clients]


class SCAFFOLDStrategy(FedAvgStrategy):
    """
    SCAFFOLD Strategy with control variates.
    
    Maintains server control variate and aggregates client control updates.
    """
    
    def __init__(self, num_clients: int = 3, **kwargs):
        super().__init__(**kwargs)
        
        self.num_clients = num_clients
        
        # Initialize server control variate to zeros
        self.server_control = [
            np.zeros_like(p) for p in get_parameters(self.model)
        ]
        
        # Track client control variates
        self.client_controls = {
            i: [np.zeros_like(p) for p in get_parameters(self.model)]
            for i in range(num_clients)
        }
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model parameters and control variates.
        
        SCAFFOLD aggregation:
        1. Model parameters: Weighted average (same as FedAvg)
        2. Control variates: c_new = c_old + (1/N) * Σ delta_c_i
        
        where delta_c_i is the change in client i's control variate
        """
        
        if not results:
            return None, {}
        
        # 1. Aggregate model parameters using standard weighted averaging
        parameters_aggregated, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # 2. Aggregate control variate updates
        # Extract delta_control from metrics if available
        n_updates = 0
        control_deltas_sum = None
        
        for _, fit_res in results:
            if hasattr(fit_res, 'metrics') and 'delta_control' in fit_res.metrics:
                # Get delta control from metrics (should be serialized)
                delta_control = fit_res.metrics['delta_control']
                
                if control_deltas_sum is None:
                    control_deltas_sum = [np.zeros_like(d) for d in delta_control]
                
                # Accumulate deltas
                for i, delta in enumerate(delta_control):
                    control_deltas_sum[i] += delta
                
                n_updates += 1
        
        # Update server control: c = c + (1/N) * Σ delta_c_i
        if n_updates > 0 and control_deltas_sum is not None:
            for i in range(len(self.server_control)):
                self.server_control[i] += control_deltas_sum[i] / n_updates
        
        # Add control variate info to metrics for monitoring
        if n_updates > 0:
            metrics['scaffold_updates'] = n_updates
            # Average magnitude of control updates (for debugging)
            avg_control_norm = np.mean([np.linalg.norm(d) for d in control_deltas_sum]) / n_updates
            metrics['avg_control_delta_norm'] = float(avg_control_norm)
        
        return parameters_aggregated, metrics


class HybridStrategy(SCAFFOLDStrategy):
    """
    Hybrid FedProx + SCAFFOLD Strategy.
    
    Combines the proximal term from FedProx with SCAFFOLD's control variates.
    Supports configurable weights for each component.
    """
    
    def __init__(
        self, 
        mu: float = 0.1, 
        fedprox_weight: float = 1.0,
        scaffold_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu
        self.fedprox_weight = fedprox_weight
        self.scaffold_weight = scaffold_weight
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients with mu parameter and combination weights for hybrid training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        config["mu"] = self.mu
        config["fedprox_weight"] = self.fedprox_weight
        config["scaffold_weight"] = self.scaffold_weight
        
        fit_ins = FitIns(parameters, config)
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, fit_ins) for client in clients]


def create_strategy(
    strategy_name: str,
    testloader: DataLoader,
    input_dim: int,
    target_mean: float,
    target_std: float,
    num_clients: int = 3,
    mu: float = 0.1,
    scaffold_lr_correction: float = 1.0,
    fedprox_weight: float = 1.0,
    scaffold_weight: float = 1.0,
    log_transform: bool = False,
    initial_parameters: Optional[Parameters] = None,
    on_fit_config_fn: Optional[Callable] = None,
    on_evaluate_config_fn: Optional[Callable] = None,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
) -> Strategy:
    """
    Factory function to create a federated learning strategy.
    
    Args:
        strategy_name: One of "fedavg", "fedprox", "scaffold", "hybrid"
        testloader: Global test data loader
        input_dim: Number of input features
        target_mean: Target mean for denormalization
        target_std: Target std for denormalization
        num_clients: Number of clients
        mu: FedProx proximal term weight
        scaffold_lr_correction: SCAFFOLD learning rate correction factor
        fedprox_weight: Weight for FedProx component in hybrid (0-1)
        scaffold_weight: Weight for SCAFFOLD component in hybrid (0-1)
        log_transform: Whether log(1+y) transformation was applied to target
        initial_parameters: Initial model parameters
        on_fit_config_fn: Function to configure fit
        on_evaluate_config_fn: Function to configure evaluate
        fraction_fit: Fraction of clients to sample for training (0.0-1.0)
        fraction_evaluate: Fraction of clients to sample for evaluation (0.0-1.0)
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum available clients to start training
    
    Returns:
        Configured Strategy instance
    """
    common_args = {
        "testloader": testloader,
        "input_dim": input_dim,
        "target_mean": target_mean,
        "target_std": target_std,
        "log_transform": log_transform,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
        "on_fit_config_fn": on_fit_config_fn,
        "on_evaluate_config_fn": on_evaluate_config_fn,
    }
    
    strategy_name = strategy_name.lower()
    
    if strategy_name == "fedavg":
        return FedAvgStrategy(**common_args)
    elif strategy_name == "fedprox":
        return FedProxStrategy(mu=mu, **common_args)
    elif strategy_name == "scaffold":
        return SCAFFOLDStrategy(num_clients=num_clients, **common_args)
    elif strategy_name == "hybrid":
        return HybridStrategy(
            mu=mu, 
            fedprox_weight=fedprox_weight,
            scaffold_weight=scaffold_weight,
            num_clients=num_clients, 
            **common_args
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
