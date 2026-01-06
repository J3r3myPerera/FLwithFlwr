"""
Adaptive FedProx Strategy

Implements FedProx with adaptive proximal term (mu) that adjusts based on:
1. Loss trajectory: Increases mu when loss increases (divergence), decreases when converging
2. Round-based decay: Optionally applies decay as training stabilizes
3. Bounded mu: Keeps mu within min/max bounds to prevent extreme values

Reference: "Adaptive Federated Optimization" - Reddi et al., ICLR 2021
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from logging import WARNING

import flwr as fl
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx


class AdaptiveFedProx(FedProx):
    """
    FedProx strategy with adaptive proximal term (mu).
    
    The mu value adapts based on training dynamics:
    - Increases when loss increases between rounds (model diverging)
    - Decreases when loss decreases (model converging)
    - Stays within configurable bounds [mu_min, mu_max]
    
    Adaptation formula:
        if loss_increased:
            mu_new = min(mu * increase_factor, mu_max)
        else:
            mu_new = max(mu * decrease_factor, mu_min)
    """
    
    def __init__(
        self,
        *,
        # Adaptive mu parameters
        initial_mu: float = 0.1,
        mu_min: float = 0.001,
        mu_max: float = 1.0,
        mu_increase_factor: float = 1.5,
        mu_decrease_factor: float = 0.9,
        loss_threshold: float = 0.0,  # Increase mu if loss increases by more than this
        warmup_rounds: int = 3,  # Don't adapt mu for first N rounds
        # Standard FedProx/FedAvg parameters
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
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
    ) -> None:
        """Initialize AdaptiveFedProx strategy.
        
        Args:
            initial_mu: Starting value for proximal term
            mu_min: Minimum allowed mu value
            mu_max: Maximum allowed mu value
            mu_increase_factor: Factor to multiply mu when loss increases
            mu_decrease_factor: Factor to multiply mu when loss decreases
            loss_threshold: Only increase mu if loss change exceeds this threshold
            warmup_rounds: Number of rounds to wait before adapting mu
            ... (standard FedProx parameters)
        """
        # Store adaptive parameters before calling parent init
        self.initial_mu = initial_mu
        self.current_mu = initial_mu
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_increase_factor = mu_increase_factor
        self.mu_decrease_factor = mu_decrease_factor
        self.loss_threshold = loss_threshold
        self.warmup_rounds = warmup_rounds
        
        # Track loss history for adaptation
        self.loss_history: List[float] = []
        self.mu_history: List[Tuple[int, float]] = [(0, initial_mu)]
        self.current_round = 0
        
        # Initialize parent FedProx with initial mu
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            proximal_mu=initial_mu,
        )
        
        # Wrap the original on_fit_config_fn to inject current mu
        self._original_on_fit_config_fn = on_fit_config_fn
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        """Configure fit with current adaptive mu value."""
        self.current_round = server_round
        
        # Update the proximal_mu attribute used by parent FedProx
        self.proximal_mu = self.current_mu
        
        # Call parent's configure_fit
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and adapt mu based on aggregated metrics."""
        
        # Call parent aggregation
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Add current mu to metrics for tracking
        metrics_aggregated["adaptive_mu"] = self.current_mu
        
        return parameters_aggregated, metrics_aggregated
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate and adapt mu based on loss."""
        
        # Call parent evaluation
        result = super().evaluate(server_round, parameters)
        
        if result is not None:
            loss, metrics = result
            
            # Track loss history
            self.loss_history.append(loss)
            
            # Adapt mu after warmup period
            if server_round > self.warmup_rounds and len(self.loss_history) >= 2:
                self._adapt_mu(server_round, loss)
            
            # Add mu tracking to metrics
            metrics["adaptive_mu"] = self.current_mu
            metrics["mu_adapted"] = server_round > self.warmup_rounds
            
            return loss, metrics
        
        return result
    
    def _adapt_mu(self, server_round: int, current_loss: float) -> None:
        """
        Adapt mu based on loss trajectory.
        
        Strategy:
        - If loss increased by more than threshold: increase mu (more regularization)
        - If loss decreased or stayed same: decrease mu (less regularization needed)
        - Uses relative loss change to account for scale differences
        """
        if len(self.loss_history) < 2:
            return
        
        prev_loss = self.loss_history[-2]
        loss_change = current_loss - prev_loss
        loss_change_ratio = loss_change / (prev_loss + 1e-8)  # Relative change
        
        old_mu = self.current_mu
        
        # Use relative change for threshold check (more robust to loss scale)
        # Also check absolute change to handle very small losses
        relative_threshold = abs(loss_change_ratio) > 0.05  # 5% relative change
        absolute_threshold = abs(loss_change) > self.loss_threshold
        
        if loss_change > 0 and (relative_threshold or absolute_threshold):
            # Loss increased significantly - increase regularization
            # Scale increase factor based on magnitude of increase
            if loss_change_ratio > 0.1:  # >10% increase
                factor = self.mu_increase_factor
            else:  # 5-10% increase
                factor = 1.0 + (self.mu_increase_factor - 1.0) * 0.5  # Half the increase
            
            self.current_mu = min(
                self.current_mu * factor,
                self.mu_max
            )
            adaptation_reason = f"loss_increased_{loss_change_ratio*100:.1f}%"
        elif loss_change < 0 and (relative_threshold or absolute_threshold):
            # Loss decreased significantly - can reduce regularization
            self.current_mu = max(
                self.current_mu * self.mu_decrease_factor,
                self.mu_min
            )
            adaptation_reason = f"loss_decreased_{abs(loss_change_ratio)*100:.1f}%"
        else:
            # Loss stable (small change) - very slight decay to allow exploration
            self.current_mu = max(
                self.current_mu * 0.995,  # Very slow decay
                self.mu_min
            )
            adaptation_reason = "loss_stable"
        
        # Record mu history
        self.mu_history.append((server_round, self.current_mu))
        
        # Log adaptation
        print(f"  [AdaptiveFedProx] Round {server_round}: "
              f"Loss change: {loss_change:.6f} ({loss_change_ratio*100:.2f}%), "
              f"mu: {old_mu:.4f} -> {self.current_mu:.4f} ({adaptation_reason})")
    
    def get_mu_history(self) -> List[Tuple[int, float]]:
        """Return the history of mu values across rounds."""
        return self.mu_history.copy()
    
    def get_final_mu(self) -> float:
        """Return the final adapted mu value."""
        return self.current_mu


class AdaptiveFedProxDivergence(AdaptiveFedProx):
    """
    Variant of AdaptiveFedProx that also considers client model divergence.
    
    Adapts mu based on:
    1. Loss trajectory (inherited from AdaptiveFedProx)
    2. Average divergence of client updates from global model
    
    When client updates diverge significantly, mu is increased to bring them
    closer to the global model.
    """
    
    def __init__(
        self,
        *,
        divergence_threshold: float = 0.1,
        divergence_weight: float = 0.5,
        **kwargs
    ) -> None:
        """
        Args:
            divergence_threshold: Threshold for considering divergence significant
            divergence_weight: Weight for divergence-based adaptation (0-1)
            **kwargs: Arguments passed to AdaptiveFedProx
        """
        super().__init__(**kwargs)
        self.divergence_threshold = divergence_threshold
        self.divergence_weight = divergence_weight
        self.divergence_history: List[float] = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate and compute divergence metric."""
        
        # Get aggregated parameters from parent
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        if parameters_aggregated is not None and len(results) > 1:
            # Compute average divergence of client updates
            aggregated_ndarrays = parameters_to_ndarrays(parameters_aggregated)
            
            divergences = []
            for client_proxy, fit_res in results:
                client_params = parameters_to_ndarrays(fit_res.parameters)
                # Compute L2 norm of difference
                divergence = sum(
                    np.linalg.norm(cp - ap)
                    for cp, ap in zip(client_params, aggregated_ndarrays)
                )
                divergences.append(divergence)
            
            avg_divergence = np.mean(divergences)
            self.divergence_history.append(avg_divergence)
            
            # Adapt mu based on divergence
            if server_round > self.warmup_rounds and avg_divergence > self.divergence_threshold:
                divergence_factor = 1 + self.divergence_weight * (avg_divergence / self.divergence_threshold - 1)
                self.current_mu = min(self.current_mu * divergence_factor, self.mu_max)
                print(f"  [AdaptiveFedProx] Divergence-based adjustment: "
                      f"avg_divergence={avg_divergence:.4f}, mu -> {self.current_mu:.4f}")
            
            metrics_aggregated["client_divergence"] = avg_divergence
        
        return parameters_aggregated, metrics_aggregated

