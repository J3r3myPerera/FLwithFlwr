"""
Adaptive Multi-Layer Mu Controller for FedProx.

This module provides adaptive computation of per-layer mu values
based on round number, layer divergence, and configurable schedules.
"""

import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class AdaptiveMuConfig:
    """Configuration for adaptive mu computation."""
    # Base layer-specific mu values
    base_layer_mus: Dict[str, float] = field(default_factory=lambda: {
        'input': 0.003,
        'hidden1': 0.008,
        'hidden2': 0.012,
        'output': 0.025
    })

    # Schedule configuration
    schedule_type: str = 'cosine'  # 'cosine', 'linear', 'warmup_decay', 'constant'
    min_schedule_factor: float = 0.3
    warmup_rounds: int = 3

    # Divergence adaptation
    use_divergence_adaptation: bool = True
    divergence_weight: float = 0.5  # How much divergence affects mu

    # Bounds
    min_mu: float = 0.001
    max_mu: float = 0.1
    mu_fallback: float = 0.01


class AdaptiveMuController:
    """
    Controller for computing adaptive per-layer mu values.

    Maintains state across rounds and computes mu values based on:
    - Round-based scheduling (decay/warmup patterns)
    - Layer divergence from previous round
    - Per-layer base mu configuration
    """

    def __init__(self, config: AdaptiveMuConfig, total_rounds: int):
        self.config = config
        self.total_rounds = total_rounds
        self.current_round = 0

        # Track layer divergences across rounds
        self.layer_divergence_history: List[Dict[str, float]] = []
        self.aggregated_divergences: Dict[str, float] = {}

        # Track computed mu values for logging
        self.mu_history: List[Dict[str, float]] = []

    def compute_schedule_factor(self, round_num: int) -> float:
        """Compute the round-based schedule factor."""
        if self.total_rounds <= 1:
            return 1.0

        if self.config.schedule_type == 'cosine':
            # Cosine decay from 1.0 to min_schedule_factor
            progress = round_num / self.total_rounds
            return (self.config.min_schedule_factor +
                    (1 - self.config.min_schedule_factor) *
                    0.5 * (1 + math.cos(math.pi * progress)))

        elif self.config.schedule_type == 'linear':
            # Linear decay from 1.0 to min_schedule_factor
            progress = round_num / self.total_rounds
            return 1.0 - (1 - self.config.min_schedule_factor) * progress

        elif self.config.schedule_type == 'warmup_decay':
            # Warmup for first few rounds, then decay
            if round_num < self.config.warmup_rounds:
                # Ramp from 0.5 to 1.0 during warmup
                return 0.5 + 0.5 * (round_num / self.config.warmup_rounds)
            else:
                # Decay after warmup
                remaining = self.total_rounds - self.config.warmup_rounds
                if remaining <= 0:
                    return 1.0
                progress = (round_num - self.config.warmup_rounds) / remaining
                return 1.0 - (1 - self.config.min_schedule_factor) * progress

        else:  # 'constant'
            return 1.0

    def compute_divergence_factor(self, layer_name: str) -> float:
        """Compute divergence-based adjustment factor for a layer."""
        if not self.config.use_divergence_adaptation or not self.aggregated_divergences:
            return 1.0

        raw_divergence = self.aggregated_divergences.get(layer_name, 1.0)

        # Apply divergence weight to dampen/amplify the effect
        weighted = 1.0 + self.config.divergence_weight * (raw_divergence - 1.0)

        # Clamp to reasonable range [0.5, 2.0]
        return max(0.5, min(2.0, weighted))

    def compute_round_layer_mus(self, round_num: int) -> Dict[str, float]:
        """
        Compute adaptive mu values for all layers for the given round.

        Args:
            round_num: Current round number (1-indexed)

        Returns:
            Dictionary mapping layer names to adaptive mu values
        """
        self.current_round = round_num
        schedule_factor = self.compute_schedule_factor(round_num)

        layer_mus = {}
        for layer_name, base_mu in self.config.base_layer_mus.items():
            divergence_factor = self.compute_divergence_factor(layer_name)

            # Compute adaptive mu
            adaptive_mu = base_mu * schedule_factor * divergence_factor

            # Clamp to bounds
            adaptive_mu = max(self.config.min_mu, min(self.config.max_mu, adaptive_mu))

            layer_mus[layer_name] = adaptive_mu

        # Store for history/logging
        self.mu_history.append({
            'round': round_num,
            'schedule_factor': schedule_factor,
            **layer_mus
        })

        return layer_mus

    def update_divergences(self, client_divergences: List[Dict[str, float]]) -> None:
        """
        Update aggregated divergences from client feedback.

        Args:
            client_divergences: List of per-client divergence dicts
        """
        if not client_divergences:
            return

        # Aggregate by averaging across clients
        layer_names = self.config.base_layer_mus.keys()
        aggregated = {}

        for layer_name in layer_names:
            values = [d.get(layer_name, 1.0) for d in client_divergences if layer_name in d]
            if values:
                aggregated[layer_name] = sum(values) / len(values)
            else:
                aggregated[layer_name] = 1.0

        self.aggregated_divergences = aggregated
        self.layer_divergence_history.append(aggregated.copy())

    def get_mu_history_summary(self) -> Dict:
        """Get summary of mu adaptation history for logging/plotting."""
        return {
            'history': self.mu_history,
            'divergence_history': self.layer_divergence_history,
            'config': {
                'schedule_type': self.config.schedule_type,
                'min_schedule_factor': self.config.min_schedule_factor,
                'use_divergence': self.config.use_divergence_adaptation,
                'divergence_weight': self.config.divergence_weight,
                'base_mus': self.config.base_layer_mus
            }
        }


def get_layer_name(param_name: str) -> str:
    """
    Map parameter name to layer name.

    Model structure: layers.0/1 (input), layers.4/5 (hidden1),
                    layers.8/9 (hidden2), layers.12 (output)
    """
    if 'layers.0' in param_name or 'layers.1' in param_name:
        return 'input'
    elif 'layers.4' in param_name or 'layers.5' in param_name:
        return 'hidden1'
    elif 'layers.8' in param_name or 'layers.9' in param_name:
        return 'hidden2'
    elif 'layers.12' in param_name:
        return 'output'
    else:
        return 'output'  # Default fallback


def compute_layer_divergences(
    local_params: List,
    global_params: List,
    param_names: List[str]
) -> Dict[str, float]:
    """
    Compute normalized divergence for each layer.

    This measures how much each layer has drifted from the global model
    after local training, normalized by the global parameter norm.

    Args:
        local_params: List of local model parameter tensors (after training)
        global_params: List of global model parameter tensors (before training)
        param_names: List of parameter names

    Returns:
        Dictionary mapping layer names to divergence factors in range [0.5, 2.0]
        - < 1.0: Layer drifted less than average
        - = 1.0: Average/baseline drift
        - > 1.0: Layer drifted more than average
    """
    # Accumulate per-layer divergence and norms
    layer_divergence = {'input': 0.0, 'hidden1': 0.0, 'hidden2': 0.0, 'output': 0.0}
    layer_norm = {'input': 0.0, 'hidden1': 0.0, 'hidden2': 0.0, 'output': 0.0}

    for idx, (local_p, global_p) in enumerate(zip(local_params, global_params)):
        if idx >= len(param_names):
            break

        layer_name = get_layer_name(param_names[idx])

        # Convert to tensors if needed
        if not isinstance(local_p, torch.Tensor):
            local_p = torch.tensor(local_p, dtype=torch.float32)
        if not isinstance(global_p, torch.Tensor):
            global_p = torch.tensor(global_p, dtype=torch.float32)

        # Ensure same device
        if local_p.device != global_p.device:
            global_p = global_p.to(local_p.device)

        diff = local_p.float() - global_p.float()
        layer_divergence[layer_name] += torch.norm(diff).item() ** 2
        layer_norm[layer_name] += torch.norm(global_p.float()).item() ** 2

    # Compute normalized divergence factors
    divergence_factors = {}
    for layer_name in layer_divergence.keys():
        if layer_norm[layer_name] > 1e-10:
            # Normalized divergence: sqrt(||local - global||^2 / ||global||^2)
            normalized = math.sqrt(layer_divergence[layer_name] / layer_norm[layer_name])
        else:
            normalized = 0.0

        # Map to factor range [0.5, 2.0]
        # Higher divergence -> higher factor -> higher mu (more regularization)
        # Scale factor determines sensitivity
        scale = 2.0  # Sensitivity parameter
        divergence_factors[layer_name] = max(0.5, min(2.0, 1.0 + normalized * scale))

    return divergence_factors
