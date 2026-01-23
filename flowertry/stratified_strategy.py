"""
Stratified FedAvg Strategy for Federated Learning.

This module extends Flower's FedAvg strategy to use stratified client selection
instead of uniform random sampling, addressing data heterogeneity while maintaining
fairness guarantees.
"""

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    FitIns,
    EvaluateIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
import numpy as np

from stratified_selector import StratifiedClientSelector


class StratifiedFedAvg(FedAvg):
    """
    FedAvg strategy with stratified client selection.
    
    Extends Flower's FedAvg to:
    1. Use stratified sampling instead of uniform random sampling
    2. Ensure balanced representation across data strata (e.g., City Tiers)
    3. Maintain fairness guarantees (minimum representation per stratum)
    
    Key benefits:
    - Reduces gradient variance through stratified sampling
    - Prevents "toxic" client combinations that cause divergence
    - Guarantees fairness across demographic groups
    - Maintains FedAvg convergence rate with tighter constants
    """
    
    def __init__(
        self,
        *args,
        stratified_selector: Optional[StratifiedClientSelector] = None,
        **kwargs
    ):
        """
        Initialize StratifiedFedAvg strategy.
        
        Args:
            stratified_selector: Optional StratifiedClientSelector for client selection.
                                If None, falls back to standard random selection.
            *args, **kwargs: Arguments passed to parent FedAvg class.
        """
        super().__init__(*args, **kwargs)
        self.stratified_selector = stratified_selector
        
        # Track metrics for analysis
        self.selection_metrics: List[Dict[str, any]] = []
        
        if self.stratified_selector is not None:
            print("\n" + "="*80)
            print("USING STRATIFIED CLIENT SELECTION")
            print("="*80)
            print("This strategy ensures balanced representation across client strata")
            print("to reduce gradient variance and prevent toxic client combinations.")
            print("="*80 + "\n")
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training with stratified client selection.
        
        Args:
            server_round: Current server round
            parameters: Current global model parameters
            client_manager: ClientManager for accessing clients
        
        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients
        """
        # Get configuration from on_fit_config_fn
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Convert parameters to FitIns
        fit_ins = FitIns(parameters, config)
        
        # Get all available clients
        # client_manager.all() returns a dict-like object with cid as keys
        all_clients_dict = client_manager.all()
        
        if self.stratified_selector is not None:
            # Use stratified selection
            # Note: Uses global numpy seed set in main() - no seed override
            selected_client_ids = self.stratified_selector.select_clients(
                round_num=server_round
            )
            
            # Map client IDs to ClientProxy objects
            # Convert selected_client_ids to strings (Flower uses string cids)
            selected_clients = []
            for cid in selected_client_ids:
                cid_str = str(cid)
                if cid_str in all_clients_dict:
                    selected_clients.append(all_clients_dict[cid_str])
            
            # Log selection
            stats = self.stratified_selector.get_stratum_statistics(selected_client_ids)
            print(f"\nRound {server_round} - Stratified Selection:")
            print(f"  Selected {len(selected_clients)} clients: {selected_client_ids}")
            print(f"  Stratum distribution: {stats}")
            
            # Store metrics
            self.selection_metrics.append({
                'round': server_round,
                'selected_clients': selected_client_ids,
                'stratum_stats': stats
            })
        else:
            # Fall back to standard random selection
            all_clients_list = list(all_clients_dict.values())
            num_clients = int(self.fraction_fit * len(all_clients_list))
            num_clients = max(num_clients, self.min_fit_clients)
            num_clients = min(num_clients, len(all_clients_list))
            selected_clients = np.random.choice(all_clients_list, num_clients, replace=False).tolist()
            
            print(f"\nRound {server_round} - Random Selection:")
            print(f"  Selected {len(selected_clients)} clients")
        
        # Return list of (ClientProxy, FitIns) tuples
        return [(client, fit_ins) for client in selected_clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation with stratified client selection.
        
        Args:
            server_round: Current server round
            parameters: Current global model parameters
            client_manager: ClientManager for accessing clients
        
        Returns:
            List of (ClientProxy, EvaluateIns) tuples for selected clients
        """
        # Don't configure federated evaluation if fraction_evaluate is 0
        if self.fraction_evaluate == 0.0:
            return []
        
        # Get configuration
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        # Convert parameters to EvaluateIns
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Get all available clients
        all_clients_dict = client_manager.all()
        
        if self.stratified_selector is not None:
            # Use stratified selection for evaluation too
            # This ensures evaluation is also balanced across strata
            # Note: Uses global numpy seed set in main() - no seed override
            selected_client_ids = self.stratified_selector.select_clients(
                round_num=server_round
            )
            
            # Map client IDs to ClientProxy objects
            selected_clients = []
            for cid in selected_client_ids:
                cid_str = str(cid)
                if cid_str in all_clients_dict:
                    selected_clients.append(all_clients_dict[cid_str])
        else:
            # Fall back to standard random selection
            all_clients_list = list(all_clients_dict.values())
            num_clients = int(self.fraction_evaluate * len(all_clients_list))
            num_clients = max(num_clients, self.min_evaluate_clients)
            num_clients = min(num_clients, len(all_clients_list))
            selected_clients = np.random.choice(all_clients_list, num_clients, replace=False).tolist()
        
        # Return list of (ClientProxy, EvaluateIns) tuples
        return [(client, evaluate_ins) for client in selected_clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results from clients.
        
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
        
        return aggregated_parameters, aggregated_metrics
    
    def get_selection_metrics(self) -> List[Dict[str, any]]:
        """Get history of client selection metrics."""
        return self.selection_metrics
    
    def get_fairness_metrics(self) -> Dict[str, float]:
        """
        Get fairness metrics from stratified selector.
        
        Returns:
            Dictionary with fairness metrics including:
            - participation_equity_gini: Gini coefficient of client participation
            - representation_ratios: Per-stratum representation quality
            - toxic_round_frequency_pct: Percentage of unbalanced rounds
        """
        if self.stratified_selector is not None:
            return self.stratified_selector.compute_fairness_metrics()
        return {}
    
    def print_final_summary(self):
        """Print final summary of stratified selection performance."""
        if self.stratified_selector is None:
            return
        
        print("\n" + "="*80)
        print("STRATIFIED SELECTION FINAL SUMMARY")
        print("="*80)
        
        fairness_metrics = self.get_fairness_metrics()
        
        print(f"\nTotal rounds: {fairness_metrics.get('total_rounds', 0)}")
        print(f"Participation equity (Gini): {fairness_metrics.get('participation_equity_gini', 0):.4f}")
        print(f"  (0 = perfect equality, 1 = maximum inequality)")
        
        print(f"\nRepresentation ratios (actual/expected):")
        ratios = fairness_metrics.get('representation_ratios', {})
        for stratum, ratio in ratios.items():
            print(f"  {stratum}: {ratio:.3f} (1.0 = perfect representation)")
        
        print(f"\nToxic round frequency: {fairness_metrics.get('toxic_round_frequency_pct', 0):.1f}%")
        print(f"  (Rounds with >20% deviation from expected stratum distribution)")
        
        print("="*80 + "\n")
