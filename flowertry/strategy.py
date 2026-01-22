"""
Custom Adaptive FedProx Strategy with divergence-based mu adaptation.

This module extends Flower's FedProx strategy to support adaptive
per-layer mu values that change dynamically based on training progress
and layer divergence feedback from clients.
"""

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx

from adaptive_mu import AdaptiveMuController


class AdaptiveFedProx(FedProx):
    """
    FedProx strategy with adaptive per-layer mu values.

    Extends Flower's FedProx to:
    1. Use an AdaptiveMuController for per-round mu computation
    2. Collect and aggregate divergence metrics from clients
    3. Update the controller with divergence feedback for next round
    """

    def __init__(
        self,
        *args,
        adaptive_controller: Optional[AdaptiveMuController] = None,
        **kwargs
    ):
        """
        Initialize AdaptiveFedProx strategy.

        Args:
            adaptive_controller: Optional AdaptiveMuController for computing
                                adaptive mu values. If None, behaves like
                                standard FedProx.
            *args, **kwargs: Arguments passed to parent FedProx class.
        """
        super().__init__(*args, **kwargs)
        self.adaptive_controller = adaptive_controller
        self.round_divergences: List[Dict[str, float]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results and extract divergence metrics.

        This method:
        1. Calls parent aggregation to combine model updates
        2. Extracts divergence metrics from client fit results
        3. Updates the adaptive controller with aggregated divergences

        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples
            failures: List of failed client results

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Extract divergence metrics from client results
        if self.adaptive_controller is not None:
            client_divergences = []

            for client_proxy, fit_res in results:
                if fit_res.metrics:
                    # Extract divergence metrics (prefixed with 'div_')
                    client_div = {}
                    for key, value in fit_res.metrics.items():
                        if key.startswith('div_'):
                            layer_name = key[4:]  # Remove 'div_' prefix
                            client_div[layer_name] = float(value)

                    if client_div:
                        client_divergences.append(client_div)

            # Update controller with aggregated divergences
            if client_divergences:
                self.adaptive_controller.update_divergences(client_divergences)
                self.round_divergences.append(
                    self.adaptive_controller.aggregated_divergences.copy()
                )

                # Log divergences for monitoring
                print(f"  Round {server_round} avg divergences: "
                      f"{self.adaptive_controller.aggregated_divergences}")

        return aggregated_parameters, aggregated_metrics

    def get_divergence_history(self) -> List[Dict[str, float]]:
        """Get history of divergence metrics across rounds."""
        return self.round_divergences

    def get_current_layer_mus(self) -> Optional[Dict[str, float]]:
        """Get the current adaptive mu values if controller is available."""
        if self.adaptive_controller is not None:
            return self.adaptive_controller.compute_round_layer_mus(
                self.adaptive_controller.current_round
            )
        return None
