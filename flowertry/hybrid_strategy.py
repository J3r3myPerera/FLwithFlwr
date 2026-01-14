"""
Hybrid FedProx-SCAFFOLD Strategy (Enhanced Version)

This strategy combines the strengths of both FedProx and SCAFFOLD with
advanced integration techniques:

Core Mechanisms:
- FedProx: Acts as a "leash" with proximal regularization to prevent clients 
  from drifting too far from the global model (magnitude control)
- SCAFFOLD: Uses control variates to correct the direction of local updates
  to align with the global objective (direction correction)

Enhanced Integration Features:
1. Sequential Activation: Pure SCAFFOLD warm-up, then gradual FedProx introduction
2. Conditional Activation: Per-client drift detection to apply appropriate mechanism
3. Dual-μ Architecture: Separate μ for raw vs SCAFFOLD-corrected components

Reference Papers:
- FedProx: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
- SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for FL" (ICML 2020)
"""

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
import numpy as np
from flwr.server.client_proxy import ClientProxy


class HybridFedProxScaffoldStrategy(fl.server.strategy.Strategy):
    """
    Enhanced Hybrid FedProx-SCAFFOLD Strategy.
    
    Key Features:
    1. Sequential Activation: SCAFFOLD warm-up (Phase 1), then gradual FedProx (Phase 2)
    2. Conditional Activation: Detect drift type per client and apply appropriate mechanism
    3. Dual-μ Architecture: Different μ for raw gradients vs SCAFFOLD-corrected components
    
    This provides robust convergence in highly heterogeneous settings by:
    - Allowing SCAFFOLD to calibrate control variates before FedProx interferes
    - Applying corrections based on actual drift patterns per client
    - Fine-grained control over how FedProx interacts with SCAFFOLD corrections
    """
    
    def __init__(
        self,
        min_fit_clients: int,
        min_available_clients: int,
        min_evaluate_clients: int,
        total_clients: int,
        # Sequential Activation Parameters
        warmup_rounds: int = 10,
        initial_mu: float = 0.001,
        mu_annealing_interval: int = 5,
        mu_annealing_factor: float = 1.5,
        max_mu: float = 0.3,
        # Dual-μ Architecture Parameters
        use_dual_mu: bool = True,
        mu_raw: float = 0.1,
        mu_corrected: float = 0.001,
        # Conditional Activation Parameters
        use_drift_detection: bool = True,
        direction_drift_threshold: float = 0.3,
        magnitude_drift_threshold: float = 2.0,
        # Legacy fixed mu
        proximal_mu: float = 0.1,
        # Standard parameters
        on_fit_config_fn=None,
        evaluate_fn=None,
        initial_parameters: Optional[Parameters] = None,
    ):
        """
        Initialize Enhanced Hybrid FedProx-SCAFFOLD strategy.
        
        Args:
            min_fit_clients: Minimum number of clients to sample for training
            min_available_clients: Minimum number of clients that must be available
            min_evaluate_clients: Minimum number of clients for evaluation
            total_clients: Total number of clients in the federation
            
            # Sequential Activation (Section 3.1)
            warmup_rounds: Phase 1 duration - pure SCAFFOLD (default: 10)
            initial_mu: Starting μ after warmup (default: 0.001)
            mu_annealing_interval: Rounds between μ increases (default: 5)
            mu_annealing_factor: Multiply μ by this each interval (default: 1.5)
            max_mu: Maximum μ value (default: 0.3)
            
            # Dual-μ Architecture (Section 3.3)
            use_dual_mu: Enable separate μ for raw/corrected components (default: True)
            mu_raw: μ for uncorrected gradient component (default: 0.1)
            mu_corrected: μ for SCAFFOLD-corrected component (default: 0.001)
            
            # Conditional Activation (Section 3.2)
            use_drift_detection: Enable per-client drift detection (default: True)
            direction_drift_threshold: Cosine distance threshold (default: 0.3)
            magnitude_drift_threshold: L2 norm ratio threshold (default: 2.0)
            
            # Legacy
            proximal_mu: Fixed FedProx coefficient if sequential disabled (default: 0.1)
            
            on_fit_config_fn: Function to generate fit config per round
            evaluate_fn: Server-side evaluation function
            initial_parameters: Initial model parameters
        """
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.total_clients = total_clients
        
        # Sequential Activation parameters
        self.warmup_rounds = warmup_rounds
        self.initial_mu = initial_mu
        self.mu_annealing_interval = mu_annealing_interval
        self.mu_annealing_factor = mu_annealing_factor
        self.max_mu = max_mu
        
        # Dual-μ Architecture parameters
        self.use_dual_mu = use_dual_mu
        self.mu_raw = mu_raw
        self.mu_corrected = mu_corrected
        
        # Conditional Activation parameters
        self.use_drift_detection = use_drift_detection
        self.direction_drift_threshold = direction_drift_threshold
        self.magnitude_drift_threshold = magnitude_drift_threshold
        
        # Legacy fixed mu (fallback)
        self.proximal_mu = proximal_mu
        self.current_mu = 0.0  # Will be updated per round
        
        self.on_fit_config_fn = on_fit_config_fn
        self.user_evaluate_fn = evaluate_fn
        self.initial_parameters = initial_parameters
        
        # Current global model parameters
        self.current_parameters: Optional[List] = None
        
        # SCAFFOLD control variates
        self.c_global: Optional[List[np.ndarray]] = None
        self.client_control_variates: Dict[str, List[np.ndarray]] = {}
        
        # Client drift tracking (for conditional activation)
        self.client_last_update: Dict[str, List[np.ndarray]] = {}
        self.global_update_direction: Optional[List[np.ndarray]] = None
        
        # Track mu adaptation history
        self.mu_history: List[Tuple[int, float]] = []
        self.phase_history: List[Tuple[int, str]] = []  # Track which phase we're in
    
    def _compute_current_mu(self, server_round: int) -> float:
        """
        Compute the current μ value based on sequential activation strategy.
        
        Phase 1 (Rounds 1 to warmup_rounds): μ = 0 (pure SCAFFOLD)
        Phase 2 (Rounds warmup_rounds+1 onwards): Gradual μ annealing
        
        Returns:
            Current μ value for this round
        """
        if server_round <= self.warmup_rounds:
            # Phase 1: SCAFFOLD warm-up, no proximal term
            return 0.0
        else:
            # Phase 2: Gradual FedProx introduction
            rounds_since_warmup = server_round - self.warmup_rounds
            # Calculate number of annealing steps
            annealing_steps = rounds_since_warmup // self.mu_annealing_interval
            # Compute annealed μ
            annealed_mu = self.initial_mu * (self.mu_annealing_factor ** annealing_steps)
            # Clamp to max_mu
            return min(annealed_mu, self.max_mu)
    
    def _compute_client_drift(self, cid: str, client_update: List[np.ndarray]) -> Tuple[float, float]:
        """
        Decompose client drift into direction and magnitude components.
        
        Direction drift: Cosine distance between local and global update directions
        Magnitude drift: L2 norm ratio of local to global update
        
        Args:
            cid: Client ID
            client_update: Client's model update (delta weights)
            
        Returns:
            (direction_drift, magnitude_drift) tuple
        """
        if self.global_update_direction is None:
            # First round, no global direction yet
            return 0.0, 1.0
        
        # Flatten updates for comparison
        client_flat = np.concatenate([u.flatten() for u in client_update])
        global_flat = np.concatenate([u.flatten() for u in self.global_update_direction])
        
        # Compute norms
        client_norm = np.linalg.norm(client_flat)
        global_norm = np.linalg.norm(global_flat)
        
        if client_norm < 1e-10 or global_norm < 1e-10:
            return 0.0, 1.0
        
        # Direction drift: 1 - cosine_similarity (0 = same direction, 2 = opposite)
        cosine_sim = np.dot(client_flat, global_flat) / (client_norm * global_norm)
        direction_drift = 1.0 - cosine_sim
        
        # Magnitude drift: ratio of norms (>1 means client update larger than global)
        magnitude_drift = client_norm / global_norm
        
        return direction_drift, magnitude_drift
    
    def _get_client_activation_mode(self, direction_drift: float, magnitude_drift: float) -> str:
        """
        Determine activation mode based on drift type.
        
        Returns one of:
        - "scaffold_only": High direction drift, low magnitude → SCAFFOLD correction only
        - "fedprox_only": High magnitude drift, low direction → FedProx only
        - "hybrid_reduced": Both drifts high → Apply hybrid with reduced μ
        - "fedavg": Both drifts low → Apply neither (effectively FedAvg)
        """
        high_direction = direction_drift > self.direction_drift_threshold
        high_magnitude = magnitude_drift > self.magnitude_drift_threshold
        
        if high_direction and not high_magnitude:
            return "scaffold_only"
        elif high_magnitude and not high_direction:
            return "fedprox_only"
        elif high_direction and high_magnitude:
            return "hybrid_reduced"
        else:
            return "fedavg"
    
    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Return initial parameters if provided."""
        if self.initial_parameters is not None:
            return self.initial_parameters
        return None
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for hybrid training with enhanced mechanisms."""
        
        # Initialize current parameters if not done
        if self.current_parameters is None:
            self.current_parameters = parameters_to_ndarrays(parameters)
        
        # Initialize global control variate c_global if not done
        if self.c_global is None:
            self.c_global = [np.zeros_like(p) for p in self.current_parameters]
        
        # Compute current μ based on sequential activation strategy
        self.current_mu = self._compute_current_mu(server_round)
        
        # Determine current phase
        if server_round <= self.warmup_rounds:
            phase = "Phase1_SCAFFOLD_Warmup"
        else:
            phase = f"Phase2_Hybrid_mu={self.current_mu:.4f}"
        self.phase_history.append((server_round, phase))
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        fit_instructions = []
        
        for client in clients:
            cid = client.cid
            
            # Initialize per-client control variate c_i if new client
            if cid not in self.client_control_variates:
                self.client_control_variates[cid] = [
                    np.zeros_like(p) for p in self.current_parameters
                ]
            
            # Build config dictionary
            config_dict = {}
            if self.on_fit_config_fn is not None:
                config_dict.update(self.on_fit_config_fn(server_round))
            
            # Determine client-specific activation mode if drift detection enabled
            activation_mode = "hybrid"  # Default
            client_mu = self.current_mu
            use_scaffold = True
            
            if self.use_drift_detection and cid in self.client_last_update:
                # We have previous update data for this client
                last_update = self.client_last_update[cid]
                direction_drift, magnitude_drift = self._compute_client_drift(cid, last_update)
                activation_mode = self._get_client_activation_mode(direction_drift, magnitude_drift)
                
                if activation_mode == "scaffold_only":
                    client_mu = 0.0
                    use_scaffold = True
                elif activation_mode == "fedprox_only":
                    client_mu = self.current_mu
                    use_scaffold = False
                elif activation_mode == "hybrid_reduced":
                    client_mu = self.current_mu * 0.5  # Reduced μ
                    use_scaffold = True
                elif activation_mode == "fedavg":
                    client_mu = 0.0
                    use_scaffold = False
            
            # Inject hybrid-specific parameters
            config_dict["hybrid"] = True  # Flag for hybrid training
            config_dict["use_scaffold"] = use_scaffold  # Whether to apply SCAFFOLD correction
            config_dict["activation_mode"] = activation_mode
            
            # Dual-μ Architecture parameters
            if self.use_dual_mu and self.current_mu > 0:
                config_dict["use_dual_mu"] = True
                config_dict["mu_raw"] = self.mu_raw
                config_dict["mu_corrected"] = self.mu_corrected
                config_dict["proximal_mu"] = 0.0  # Disable legacy single-mu
            else:
                config_dict["use_dual_mu"] = False
                config_dict["proximal_mu"] = client_mu  # Sequential activation mu
            
            # SCAFFOLD control variates
            config_dict["c_global"] = self.c_global
            config_dict["c_local"] = self.client_control_variates[cid]
            
            fit_ins = FitIns(parameters, config_dict)
            fit_instructions.append((client, fit_ins))
        
        # Track mu history
        self.mu_history.append((server_round, self.current_mu))
        
        return fit_instructions
        
        return fit_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates and control variates with drift tracking."""
        
        if not results:
            return None, {}
        
        deltas_w = []
        delta_cs = []
        num_examples = []
        
        for client, fit_res in results:
            cid = client.cid
            if fit_res.num_examples == 0:
                continue
            
            # Model delta: w_i - w_global
            w_i = parameters_to_ndarrays(fit_res.parameters)
            delta_w = [w_i[j] - self.current_parameters[j] for j in range(len(w_i))]
            deltas_w.append(delta_w)
            num_examples.append(fit_res.num_examples)
            
            # Store client update for drift detection in next round
            self.client_last_update[cid] = delta_w
            
            # Control variate delta from client
            if "delta_c" not in fit_res.metrics:
                raise RuntimeError(f"Client {cid} did not return delta_c for hybrid strategy")
            
            delta_c = fit_res.metrics["delta_c"]
            # Ensure each delta_c element is numpy array
            delta_c = [
                np.array(dc) if not isinstance(dc, np.ndarray) else dc
                for dc in delta_c
            ]
            
            if len(delta_c) != len(self.c_global):
                raise ValueError(f"Invalid delta_c dimension from client {cid}")
            
            delta_cs.append(delta_c)
            
            # Update local c_i
            self.client_control_variates[cid] = [
                self.client_control_variates[cid][j] + delta_c[j]
                for j in range(len(delta_c))
            ]
        
        # Aggregate model updates (weighted average)
        total_examples = sum(num_examples)
        avg_delta_w = [
            sum(deltas_w[i][j] * num_examples[i] for i in range(len(deltas_w)))
            / total_examples
            for j in range(len(deltas_w[0]))
        ]
        
        # Store global update direction for drift detection
        self.global_update_direction = avg_delta_w
        
        # Update global model
        self.current_parameters = [
            self.current_parameters[j] + avg_delta_w[j] for j in range(len(avg_delta_w))
        ]
        
        # Aggregate control variates (simple average over participating clients)
        avg_delta_c = [
            sum(delta_cs[i][j] for i in range(len(delta_cs))) / len(delta_cs)
            for j in range(len(self.c_global))
        ]
        
        # Update global control variate
        self.c_global = [
            self.c_global[j] + avg_delta_c[j] for j in range(len(self.c_global))
        ]
        
        # Compute metrics
        c_global_norm = np.sqrt(sum(np.sum(x * x) for x in self.c_global))
        
        # Determine current phase for reporting
        if server_round <= self.warmup_rounds:
            phase = "Phase1_SCAFFOLD"
        else:
            phase = "Phase2_Hybrid"
        
        metrics = {
            "c_global_norm": c_global_norm,
            "current_mu": self.current_mu,
            "phase": phase,
            "warmup_rounds": self.warmup_rounds,
            "use_dual_mu": int(self.use_dual_mu),
            "use_drift_detection": int(self.use_drift_detection),
            "num_clients": len(results),
        }
        
        # Add dual-μ values if enabled
        if self.use_dual_mu:
            metrics["mu_raw"] = self.mu_raw
            metrics["mu_corrected"] = self.mu_corrected
        
        return ndarrays_to_parameters(self.current_parameters), metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        """Configure evaluation (not used - we use server-side evaluation)."""
        return []
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        return None, {}
    
    def evaluate(self, server_round: int, parameters: Parameters):
        """Server-side evaluation."""
        if self.user_evaluate_fn is not None:
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            return self.user_evaluate_fn(server_round, parameters_ndarrays, {})
        return None
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients to sample for training."""
        return self.min_fit_clients, self.min_available_clients
    
    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return number of clients to sample for evaluation."""
        return self.min_evaluate_clients, self.min_available_clients
    
    def get_mu_history(self) -> List[Tuple[int, float]]:
        """Get history of mu values."""
        return self.mu_history
    
    def get_phase_history(self) -> List[Tuple[int, str]]:
        """Get history of training phases."""
        return self.phase_history
    
    def get_control_variates(self, client_id: str):
        """Get control variates for a specific client."""
        c_i = self.client_control_variates.get(client_id, None)
        return self.c_global, c_i
    
    def get_config_summary(self) -> Dict:
        """Get summary of configuration for logging."""
        return {
            "warmup_rounds": self.warmup_rounds,
            "initial_mu": self.initial_mu,
            "mu_annealing_interval": self.mu_annealing_interval,
            "mu_annealing_factor": self.mu_annealing_factor,
            "max_mu": self.max_mu,
            "use_dual_mu": self.use_dual_mu,
            "mu_raw": self.mu_raw if self.use_dual_mu else None,
            "mu_corrected": self.mu_corrected if self.use_dual_mu else None,
            "use_drift_detection": self.use_drift_detection,
            "direction_drift_threshold": self.direction_drift_threshold if self.use_drift_detection else None,
            "magnitude_drift_threshold": self.magnitude_drift_threshold if self.use_drift_detection else None,
        }
