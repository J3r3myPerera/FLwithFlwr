"""
FedProx-SCAFFOLD Hybrid Strategy Implementation for Flower.

This strategy combines:
- FedProx: Proximal term regularization to keep local models close to global model
- SCAFFOLD: Control variates for variance reduction and client-drift correction

The hybrid approach addresses data heterogeneity from two angles:
1. Proximal term prevents local models from drifting too far (FedProx)
2. Control variates correct for the direction of drift (SCAFFOLD)

This combination is particularly effective for:
- Highly heterogeneous (non-IID) data distributions
- Settings where both convergence stability and accuracy are important
- Personal finance modeling where client data can vary significantly
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate


class FedProxScaffoldStrategy(Strategy):
    """
    Hybrid FedProx-SCAFFOLD Strategy.
    
    Combines FedProx's proximal regularization with SCAFFOLD's control variates
    for improved convergence on heterogeneous data.
    
    Key features:
    - Proximal term: (μ/2) * ||w - w_global||² added to local objective
    - Control variates: c_global and c_client for variance reduction
    - Adaptive combination based on heterogeneity level
    
    Papers:
    - FedProx: "Federated Optimization in Heterogeneous Networks" (Li et al., 2018)
    - SCAFFOLD: "Stochastic Controlled Averaging for FL" (Karimireddy et al., 2020)
    """
    
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        # FedProx parameters
        proximal_mu: float = 0.1,
        # SCAFFOLD parameters
        scaffold_lr: float = 1.0,
        # Hybrid parameters
        adaptive_weights: bool = True,
        prox_weight: float = 0.5,  # Weight for proximal term (1-prox_weight for SCAFFOLD)
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        
        # FedProx parameters
        self.proximal_mu = proximal_mu
        
        # SCAFFOLD parameters
        self.scaffold_lr = scaffold_lr
        
        # Hybrid parameters
        self.adaptive_weights = adaptive_weights
        self.prox_weight = prox_weight
        
        # SCAFFOLD state: Global control variate
        self.c_global: Optional[NDArrays] = None
        # Client control variates (stored per client)
        self.c_client: Dict[int, NDArrays] = {}
        
        # Tracking for adaptive weighting
        self.round_losses: List[float] = []
        self.round_accuracies: List[float] = []
    
    def __repr__(self) -> str:
        return f"FedProxScaffoldStrategy(mu={self.proximal_mu}, scaffold_lr={self.scaffold_lr})"
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters
    
    def _compute_adaptive_weights(self, server_round: int) -> Tuple[float, float]:
        """
        Compute adaptive weights for proximal and SCAFFOLD terms.
        
        Early rounds: Higher proximal weight (stability)
        Later rounds: Higher SCAFFOLD weight (variance reduction)
        
        Returns:
            (prox_weight, scaffold_weight)
        """
        if not self.adaptive_weights:
            return self.prox_weight, 1.0 - self.prox_weight
        
        # Adaptive schedule: start with more proximal, shift to SCAFFOLD
        warmup_rounds = 5
        if server_round <= warmup_rounds:
            # Early rounds: prioritize proximal term for stability
            prox_w = 0.7
        elif server_round <= 2 * warmup_rounds:
            # Transition phase
            progress = (server_round - warmup_rounds) / warmup_rounds
            prox_w = 0.7 - 0.3 * progress  # 0.7 -> 0.4
        else:
            # Later rounds: prioritize SCAFFOLD for variance reduction
            prox_w = 0.4
        
        # Check if we're seeing oscillations in loss (sign of instability)
        if len(self.round_losses) >= 3:
            recent_losses = self.round_losses[-3:]
            # If loss is increasing, boost proximal term
            if recent_losses[-1] > recent_losses[-2] > recent_losses[-3]:
                prox_w = min(0.8, prox_w + 0.2)
        
        return prox_w, 1.0 - prox_w
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Initialize c_global if not done yet
        if self.c_global is None and parameters is not None:
            self.c_global = [np.zeros_like(arr) for arr in parameters_to_ndarrays(parameters)]
        
        # Compute adaptive weights
        prox_weight, scaffold_weight = self._compute_adaptive_weights(server_round)
        
        # Add hybrid parameters to config
        config["is_hybrid"] = True
        config["proximal_mu"] = self.proximal_mu * prox_weight
        config["scaffold_lr"] = self.scaffold_lr * scaffold_weight
        config["prox_weight"] = prox_weight
        config["scaffold_weight"] = scaffold_weight
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Prepare config for each client
        fit_configurations = []
        for client in clients:
            # Get or initialize client control variate
            client_id = id(client)
            if client_id not in self.c_client:
                if parameters is not None:
                    self.c_client[client_id] = [np.zeros_like(arr) for arr in parameters_to_ndarrays(parameters)]
                else:
                    self.c_client[client_id] = None
            
            # Add control variates to config
            client_config = config.copy()
            if self.c_global is not None:
                client_config["c_global"] = self.c_global
            if self.c_client[client_id] is not None:
                client_config["c_client"] = self.c_client[client_id]
            
            # Create FitIns with parameters and config
            fit_ins = FitIns(parameters, client_config)
            fit_configurations.append((client, fit_ins))
        
        return fit_configurations
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        
        if failures and not self.accept_failures:
            return None, {}
        
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate model parameters
        aggregated_weights = aggregate(weights_results)
        
        # Update control variates from client updates
        c_updates = []
        total_samples = 0
        
        for _, fit_res in results:
            if "c_update" in fit_res.metrics:
                c_update = fit_res.metrics["c_update"]
                num_examples = fit_res.num_examples
                c_updates.append((c_update, num_examples))
                total_samples += num_examples
        
        # Update global control variate (SCAFFOLD aggregation)
        if c_updates and self.c_global is not None:
            for i in range(len(self.c_global)):
                weighted_update = np.zeros_like(self.c_global[i])
                for c_update, num_examples in c_updates:
                    if c_update is not None and len(c_update) > i:
                        weighted_update += c_update[i] * (num_examples / total_samples)
                # Update with momentum for stability
                self.c_global[i] = 0.9 * self.c_global[i] + 0.1 * weighted_update
        
        # Update client control variates
        for client_proxy, fit_res in results:
            client_id = id(client_proxy)
            if "c_client_new" in fit_res.metrics:
                self.c_client[client_id] = fit_res.metrics["c_client_new"]
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        
        # Aggregate custom metrics
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif results:
            for _, fit_res in results:
                for key, value in fit_res.metrics.items():
                    if key not in ["c_update", "c_client_new"]:
                        if isinstance(value, (int, float)):
                            if key not in metrics_aggregated:
                                metrics_aggregated[key] = []
                            metrics_aggregated[key].append(value)
            
            for key in metrics_aggregated:
                if isinstance(metrics_aggregated[key], list):
                    metrics_aggregated[key] = sum(metrics_aggregated[key]) / len(metrics_aggregated[key])
        
        return parameters_aggregated, metrics_aggregated
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, config) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        
        if failures and not self.accept_failures:
            return None, {}
        
        # Aggregate loss
        total_examples = sum([res.num_examples for _, res in results])
        weighted_loss = sum([res.num_examples * res.loss for _, res in results])
        loss_aggregated = weighted_loss / total_examples if total_examples > 0 else 0.0
        
        # Track loss for adaptive weighting
        self.round_losses.append(loss_aggregated)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif results:
            for _, evaluate_res in results:
                for key, value in evaluate_res.metrics.items():
                    if isinstance(value, (int, float)):
                        if key not in metrics_aggregated:
                            metrics_aggregated[key] = []
                        metrics_aggregated[key].append(value)
            
            for key in metrics_aggregated:
                if isinstance(metrics_aggregated[key], list):
                    metrics_aggregated[key] = sum(metrics_aggregated[key]) / len(metrics_aggregated[key])
        
        return loss_aggregated, metrics_aggregated
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        loss, metrics = eval_res
        
        # Track for adaptive weighting
        self.round_losses.append(loss)
        if "accuracy" in metrics:
            self.round_accuracies.append(metrics["accuracy"])
        
        return loss, metrics
