"""
Stratified Client Selection for Federated Learning.

This module implements stratified client selection to address data heterogeneity
in federated learning by ensuring balanced representation across client strata
(e.g., City Tiers in personal finance data).

Based on classical stratified sampling theory adapted to FL context.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StratifiedSelectionStats:
    """Statistics for a single round of stratified selection."""
    round_num: int
    selected_clients: List[int]
    stratum_counts: Dict[str, int]
    stratum_percentages: Dict[str, float]


class StratifiedClientSelector:
    """
    Ensures balanced representation across client strata.
    
    Research gap: Adapting stratified sampling theory to FL client selection
    with provable convergence guarantees and fairness properties.
    
    Key features:
    - Proportional allocation: K_h ∝ N_h (stratum size)
    - Minimum guarantee: K_h ≥ 1 for all strata (fairness)
    - Budget constraint: Σ K_h = K (total clients per round)
    """
    
    def __init__(
        self,
        client_strata: Dict[str, List[int]],
        clients_per_round: int = 5,
        min_per_stratum: int = 1
    ):
        """
        Initialize stratified client selector.
        
        Args:
            client_strata: Dictionary mapping stratum names to client IDs
                Example: {'Tier_1': [0,1,2,3],
                         'Tier_2': [4,5,6,7,8,9],
                         'Tier_3': [10,11,12,13]}
            clients_per_round: Total clients to select per round (K)
            min_per_stratum: Minimum clients to select from each stratum
        """
        self.strata = client_strata
        self.k = clients_per_round
        self.min_per_stratum = min_per_stratum
        
        # Compute stratum sizes for proportional allocation
        self.strata_sizes = {s: len(clients) for s, clients in client_strata.items()}
        self.total_clients = sum(self.strata_sizes.values())
        self.num_strata = len(self.strata)
        
        # Validate configuration
        if self.k < self.num_strata * self.min_per_stratum:
            raise ValueError(
                f"clients_per_round ({self.k}) must be >= "
                f"num_strata ({self.num_strata}) * min_per_stratum ({self.min_per_stratum})"
            )
        
        # Compute base allocation per stratum (Neyman allocation)
        self.base_allocation = self._compute_base_allocation()
        
        # Track selection history for analysis
        self.selection_history: List[StratifiedSelectionStats] = []
        
        print(f"\n{'='*80}")
        print("STRATIFIED CLIENT SELECTOR INITIALIZED")
        print(f"{'='*80}")
        print(f"Total clients: {self.total_clients}")
        print(f"Number of strata: {self.num_strata}")
        print(f"Clients per round: {self.k}")
        print(f"Minimum per stratum: {self.min_per_stratum}")
        print(f"\nStratum sizes:")
        for stratum, size in self.strata_sizes.items():
            print(f"  {stratum}: {size} clients ({size/self.total_clients*100:.1f}%)")
        print(f"\nBase allocation per round:")
        for stratum, count in self.base_allocation.items():
            print(f"  {stratum}: {count} clients")
        print(f"{'='*80}\n")
    
    def _compute_base_allocation(self) -> Dict[str, int]:
        """
        Compute base allocation of clients per stratum.
        
        Uses proportional allocation with minimum guarantee:
        - Each stratum gets at least min_per_stratum clients
        - Remaining slots allocated proportionally to stratum size
        """
        allocation = {}
        
        # Phase 1: Allocate minimum to each stratum
        remaining_slots = self.k - (self.num_strata * self.min_per_stratum)
        
        if remaining_slots < 0:
            # Not enough slots for minimum guarantee - distribute equally
            for stratum in self.strata.keys():
                allocation[stratum] = max(1, self.k // self.num_strata)
        else:
            # Phase 2: Distribute remaining slots proportionally
            # Use fractional allocation and round intelligently
            strata_list = list(self.strata_sizes.keys())
            fractional_alloc = {}
            
            for stratum, size in self.strata_sizes.items():
                # Start with minimum
                base = self.min_per_stratum
                
                # Add proportional share of remaining slots (keep as float)
                proportion = size / self.total_clients
                fractional_alloc[stratum] = base + (remaining_slots * proportion)
            
            # Round down to get initial allocation
            for stratum in strata_list:
                allocation[stratum] = int(fractional_alloc[stratum])
            
            # Distribute remaining slots based on fractional parts
            current_sum = sum(allocation.values())
            deficit = self.k - current_sum
            
            if deficit > 0:
                # Sort strata by fractional part (largest first)
                fractional_parts = {
                    s: fractional_alloc[s] - allocation[s] 
                    for s in strata_list
                }
                sorted_strata = sorted(
                    fractional_parts.keys(), 
                    key=lambda s: fractional_parts[s], 
                    reverse=True
                )
                
                # Give extra slots to strata with largest fractional parts
                for i in range(deficit):
                    allocation[sorted_strata[i]] += 1
        
        # Ensure no stratum gets more clients than it has
        for stratum in allocation.keys():
            allocation[stratum] = min(allocation[stratum], self.strata_sizes[stratum])
        
        return allocation
    
    def select_clients(self, round_num: int, seed: Optional[int] = None) -> List[int]:
        """
        Select K clients using stratified sampling.
        
        Uses the global numpy random state (set via np.random.seed() in main).
        This ensures consistent behavior with standard FedAvg random selection.
        
        Args:
            round_num: Current training round (for logging/analysis)
            seed: Optional seed (DEPRECATED - uses global numpy seed instead)
        
        Returns:
            List of selected client IDs
        """
        # Note: We deliberately DON'T override the seed here
        # This allows the global seed set in main() to control all randomness
        # consistently across both random and stratified selection
        # if seed is not None:
        #     np.random.seed(seed)  # REMOVED: Was causing unfair comparison
        
        selected = []
        stratum_counts = {}
        
        # Phase 1: Allocate base number from each stratum
        for stratum, client_ids in self.strata.items():
            n_from_stratum = self.base_allocation[stratum]
            
            # Random selection without replacement within stratum
            if n_from_stratum > 0 and len(client_ids) > 0:
                n_select = min(n_from_stratum, len(client_ids))
                stratum_selected = np.random.choice(
                    client_ids,
                    size=n_select,
                    replace=False
                )
                selected.extend(stratum_selected.tolist())
                stratum_counts[stratum] = n_select
            else:
                stratum_counts[stratum] = 0
        
        # Phase 2: Fill remaining slots proportionally if needed
        while len(selected) < self.k:
            # Sample stratum with probability proportional to size
            strata_list = list(self.strata.keys())
            probs = np.array([self.strata_sizes[s] for s in strata_list])
            probs = probs / probs.sum()
            
            stratum = np.random.choice(strata_list, p=probs)
            
            # Sample client not yet selected
            available = [c for c in self.strata[stratum] if c not in selected]
            if available:
                selected.append(np.random.choice(available))
                stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
            else:
                # All clients from this stratum already selected
                # Try another stratum
                continue
        
        # Ensure exactly k clients selected
        selected = selected[:self.k]
        
        # Compute statistics
        stratum_percentages = {
            s: (count / len(selected) * 100) if len(selected) > 0 else 0.0
            for s, count in stratum_counts.items()
        }
        
        stats = StratifiedSelectionStats(
            round_num=round_num,
            selected_clients=selected,
            stratum_counts=stratum_counts,
            stratum_percentages=stratum_percentages
        )
        self.selection_history.append(stats)
        
        return selected
    
    def get_stratum_statistics(self, selected_clients: List[int]) -> Dict[str, int]:
        """
        Compute participation statistics for given client selection.
        
        Args:
            selected_clients: List of selected client IDs
        
        Returns:
            Dictionary with per-stratum selection counts
        """
        stats = {s: 0 for s in self.strata.keys()}
        
        for client_id in selected_clients:
            for stratum, client_ids in self.strata.items():
                if client_id in client_ids:
                    stats[stratum] += 1
                    break
        
        return stats
    
    def get_selection_history(self) -> List[StratifiedSelectionStats]:
        """Get history of all client selections."""
        return self.selection_history
    
    def compute_fairness_metrics(self) -> Dict[str, float]:
        """
        Compute fairness metrics across all rounds.
        
        Returns:
            Dictionary with fairness metrics:
            - participation_equity: Gini coefficient of client selection counts
            - representation_ratio: How close each stratum's participation is to ideal
            - toxic_round_frequency: Percentage of rounds with unbalanced selection
        """
        if not self.selection_history:
            return {}
        
        # Count total selections per client
        client_selections = {i: 0 for i in range(self.total_clients)}
        for stats in self.selection_history:
            for client_id in stats.selected_clients:
                client_selections[client_id] += 1
        
        # Compute Gini coefficient for participation equity
        selection_counts = sorted(client_selections.values())
        n = len(selection_counts)
        if n == 0:
            gini = 0.0
        else:
            cumsum = np.cumsum(selection_counts)
            gini = (2 * np.sum((np.arange(1, n + 1) * selection_counts))) / (n * cumsum[-1]) - (n + 1) / n
        
        # Compute representation ratios per stratum
        representation_ratios = {}
        for stratum in self.strata.keys():
            expected_ratio = self.strata_sizes[stratum] / self.total_clients
            actual_selections = sum(stats.stratum_counts.get(stratum, 0) 
                                   for stats in self.selection_history)
            total_selections = len(self.selection_history) * self.k
            actual_ratio = actual_selections / total_selections if total_selections > 0 else 0
            representation_ratios[stratum] = actual_ratio / expected_ratio if expected_ratio > 0 else 1.0
        
        # Compute toxic round frequency (rounds where any stratum is underrepresented by >20%)
        toxic_rounds = 0
        for stats in self.selection_history:
            for stratum in self.strata.keys():
                expected_pct = (self.strata_sizes[stratum] / self.total_clients) * 100
                actual_pct = stats.stratum_percentages.get(stratum, 0.0)
                if abs(actual_pct - expected_pct) > 20:
                    toxic_rounds += 1
                    break
        
        toxic_frequency = (toxic_rounds / len(self.selection_history) * 100) if self.selection_history else 0.0
        
        return {
            'participation_equity_gini': gini,
            'representation_ratios': representation_ratios,
            'toxic_round_frequency_pct': toxic_frequency,
            'total_rounds': len(self.selection_history)
        }
    
    def print_round_summary(self, round_num: int):
        """Print summary of client selection for a specific round."""
        if round_num <= 0 or round_num > len(self.selection_history):
            return
        
        stats = self.selection_history[round_num - 1]
        print(f"\nRound {stats.round_num} Stratified Selection:")
        print(f"  Selected clients: {stats.selected_clients}")
        print(f"  Stratum distribution:")
        for stratum in sorted(self.strata.keys()):
            count = stats.stratum_counts.get(stratum, 0)
            pct = stats.stratum_percentages.get(stratum, 0.0)
            expected_pct = (self.strata_sizes[stratum] / self.total_clients) * 100
            print(f"    {stratum}: {count} clients ({pct:.1f}% vs {expected_pct:.1f}% expected)")


def create_stratified_client_fn(
    client_strata: Dict[str, List[int]],
    clients_per_round: int,
    min_per_stratum: int = 1
) -> StratifiedClientSelector:
    """
    Factory function for creating stratified selector.
    
    Usage:
        selector = create_stratified_client_fn(client_strata, K)
        # In Flower strategy
        def configure_fit(self, server_round, parameters, client_manager):
            selected_ids = selector.select_clients(server_round)
            # ... create FitIns for selected clients
    
    Args:
        client_strata: Dictionary mapping stratum names to client IDs
        clients_per_round: Number of clients to select per round
        min_per_stratum: Minimum clients per stratum
    
    Returns:
        Initialized StratifiedClientSelector
    """
    selector = StratifiedClientSelector(
        client_strata=client_strata,
        clients_per_round=clients_per_round,
        min_per_stratum=min_per_stratum
    )
    return selector
