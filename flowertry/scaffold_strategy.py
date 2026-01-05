"""
FedSCAFFOLD Strategy Implementation for Flower.

FedSCAFFOLD (SCAFFOLD) addresses client-drift in federated learning by using
control variates to correct for the difference between local and global updates.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    EvaluateRes,
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


class FedScaffoldStrategy(Strategy):
    """
    FedSCAFFOLD strategy implementation.
    
    Paper: "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    by Karimireddy et al., 2020
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
        scaffold_lr: float = 1.0,
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
        self.scaffold_lr = scaffold_lr
        
        # Global control variate (server-side)
        self.c_global: Optional[NDArrays] = None
        # Client control variates (stored per client)
        self.c_client: Dict[int, NDArrays] = {}
    
    def __repr__(self) -> str:
        return "FedScaffoldStrategy"
    
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
        if self.initial_parameters is not None:
            return self.initial_parameters
        
        # Initialize control variates to zero
        # We'll initialize them when we get the first model parameters
        return self.initial_parameters
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add scaffold_lr to config
        config["scaffold_lr"] = self.scaffold_lr
        
        # Initialize c_global if not done yet
        if self.c_global is None and parameters is not None:
            # Initialize to zero (same shape as parameters)
            self.c_global = [np.zeros_like(arr) for arr in parameters_to_ndarrays(parameters)]
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Prepare config for each client
        fit_configurations = []
        for idx, client in enumerate(clients):
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
                # Convert c_global to a format that can be sent
                # We'll send it as a list that the client can reconstruct
                client_config["c_global"] = self.c_global
            if self.c_client[client_id] is not None:
                client_config["c_client"] = self.c_client[client_id]
            
            fit_configurations.append((client, client_config))
        
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
        
        # Check for failures
        if failures and not self.accept_failures:
            return None, {}
        
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Aggregate model parameters
        aggregated_weights = aggregate(weights_results)
        
        # Update global control variate
        # In SCAFFOLD, we update c_global based on client updates
        # For simplicity, we use a simple averaging approach
        # More sophisticated implementations can use the exact SCAFFOLD update rule
        
        # Extract control variate updates from results if available
        c_updates = []
        total_samples = 0
        
        for _, fit_res in results:
            if "c_update" in fit_res.metrics:
                c_update = fit_res.metrics["c_update"]
                num_examples = fit_res.num_examples
                c_updates.append((c_update, num_examples))
                total_samples += num_examples
        
        # Update global control variate
        if c_updates and self.c_global is not None:
            # Weighted average of control variate updates
            for i in range(len(self.c_global)):
                self.c_global[i] = np.zeros_like(self.c_global[i])
                for c_update, num_examples in c_updates:
                    if c_update is not None and len(c_update) > i:
                        self.c_global[i] += c_update[i] * (num_examples / total_samples)
        
        # Update client control variates
        for client_proxy, fit_res in results:
            client_id = id(client_proxy)
            if "c_client_new" in fit_res.metrics:
                self.c_client[client_id] = fit_res.metrics["c_client_new"]
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        
        # Aggregate custom metrics if provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif results:
            # Simple average of metrics
            for _, fit_res in results:
                for key, value in fit_res.metrics.items():
                    if key not in ["c_update", "c_client_new"]:  # Skip internal metrics
                        if key not in metrics_aggregated:
                            metrics_aggregated[key] = []
                        metrics_aggregated[key].append(value)
            
            # Average the metrics
            for key in metrics_aggregated:
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
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        
        # Aggregate metrics
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif results:
            # Simple average
            for _, evaluate_res in results:
                for key, value in evaluate_res.metrics.items():
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = []
                    metrics_aggregated[key].append(value)
            
            for key in metrics_aggregated:
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
        return loss, metrics


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_sum = sum([num_examples * loss for num_examples, loss in results])
    return weighted_sum / total_examples if total_examples > 0 else 0.0

