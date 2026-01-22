"""
Federated Learning Client - v6 Decoupled SCAFFOLD Hybrid Implementation
========================================================================

CRITICAL FIX (v6): SCAFFOLD corrections are now DECOUPLED from the optimizer.
- SCAFFOLD corrections applied AFTER optimizer.step() as parameter updates
- This preserves Adam's momentum/variance estimates
- Results in more stable training and better convergence

Key Features:

1. DECOUPLED SCAFFOLD (v6 - Critical Fix)
   - SCAFFOLD corrections no longer modify gradients before optimizer.step()
   - Instead, corrections are applied as direct parameter updates AFTER optimizer.step()
   - This preserves Adam's internal state (momentum, variance estimates)

2. THREE-PHASE SCHEDULING
   - Warmup (rounds 1-3): Gentle start, minimal corrections
   - Exploration (3-30%): SCAFFOLD dominant (0.8), minimal FedProx (0.15)
   - Transition (30-70%): Gradual shift to balanced
   - Convergence (70-100%): FedProx increases for stability

3. PER-CLIENT ADAPTIVE μ
   - Clients with high drift get more FedProx regularization
   - Conservative scaling range (0.7x-1.5x)

4. GRADIENT VARIANCE-BASED SCAFFOLD
   - High gradient variance → stronger SCAFFOLD correction
   - Tighter cap (1.2x) for stability

5. LAYER-WISE CORRECTIONS
   - Early layers (features): Less correction (0.85x)
   - Later layers (task-specific): More correction (1.15x)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import flwr as fl
from flwr.common import NDArrays, Scalar

from model import DisposableIncomeNet, get_parameters, set_parameters


class RegressionClient(fl.client.NumPyClient):
    """
    Advanced Flower client with v3 hybrid implementation.
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
        mu: float = 0.1,
        scaffold_lr_correction: float = 1.0,
        fedprox_weight: float = 1.0,
        scaffold_weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
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
        self.max_grad_norm = 1.0
        
        # v3: Track client participation for staleness handling
        self.last_participated_round = 0
        self.control_staleness = 0
        
        # v3: Track gradient history for variance computation
        self.gradient_norm_history = []
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        self.model = DisposableIncomeNet(input_dim=input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        
        self.client_control = None
        self.server_control = None
        self.global_params = None
        
        # v3: Flag for control variate initialization
        self.control_initialized_from_gradients = False
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)
    
    def set_parameters(self, parameters: NDArrays) -> None:
        set_parameters(self.model, parameters)
        self.global_params = [p.clone().detach() for p in self.model.parameters()]
    
    def _init_control_variates(self):
        self.client_control = [torch.zeros_like(p) for p in self.model.parameters()]
        self.server_control = [torch.zeros_like(p) for p in self.model.parameters()]
    
    def _warm_start_control_variates(self):
        """
        v3 IMPROVEMENT: Initialize control variates from actual gradients
        instead of zeros for faster convergence.
        """
        if self.control_initialized_from_gradients:
            return
        
        self.model.train()
        gradient_sum = [torch.zeros_like(p) for p in self.model.parameters()]
        num_batches = 0
        
        # Compute average gradient over training data
        for X_batch, y_batch in self.trainloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.model.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            
            for i, p in enumerate(self.model.parameters()):
                if p.grad is not None:
                    gradient_sum[i] += p.grad.clone()
            num_batches += 1
            
            # Only use first few batches for efficiency
            if num_batches >= 5:
                break
        
        # Initialize client control to average gradient
        if self.client_control is not None:
            for i in range(len(self.client_control)):
                self.client_control[i] = (gradient_sum[i] / num_batches).to(self.device)
        
        self.control_initialized_from_gradients = True
    
    def _compute_client_divergence(self) -> float:
        """Compute L2 distance between local and global parameters."""
        if self.global_params is None:
            return 1.0
        
        divergence = sum(
            torch.sum((p - g.to(self.device)) ** 2).item()
            for p, g in zip(self.model.parameters(), self.global_params)
        )
        return np.sqrt(divergence)
    
    def _train_fedavg(self) -> float:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_fedprox(self) -> float:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
                mse_loss = self.criterion(y_pred, y_batch)
                
                proximal_term = 0.0
                if self.global_params is not None:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p.to(self.device)) ** 2)
                    proximal_term = (self.mu / 2.0) * proximal_term
                
                loss = mse_loss + proximal_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_scaffold(self) -> Tuple[float, List[np.ndarray]]:
        if self.client_control is None or self.server_control is None:
            self._init_control_variates()
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(self.trainloader) * self.local_epochs, eta_min=self.learning_rate * 0.1
        )
        
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        
        total_loss = 0.0
        n_batches = 0
        total_steps = 0
        
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for param, c_i, c in zip(self.model.parameters(), 
                                              self.client_control, 
                                              self.server_control):
                        if param.grad is not None:
                            param.grad.add_(c.to(self.device) - c_i.to(self.device))
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
                total_steps += 1
        
        old_client_control = [c.clone() for c in self.client_control]
        
        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control, 
                initial_params, self.model.parameters()
            )):
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / total_steps
                self.client_control[i] = c_i.to(self.device) - c.to(self.device) + param_diff
        
        delta_control = [
            (new_c - old_c).cpu().numpy() 
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]
        
        return total_loss / n_batches if n_batches > 0 else 0.0, delta_control
    
    def _train_hybrid(self, config: Dict[str, Scalar] = None) -> Tuple[float, List[np.ndarray]]:
        """
        v6 DECOUPLED SCAFFOLD HYBRID: Critical fix for Adam compatibility

        KEY FIX (v6): SCAFFOLD corrections applied AFTER optimizer.step()
        - This preserves Adam's momentum/variance estimates
        - SCAFFOLD corrections are now parameter updates, not gradient modifications
        - Results in more stable training and better convergence

        Features:
        1. Three-phase scheduling (exploration → transition → convergence)
        2. Per-client adaptive μ based on divergence
        3. Gradient variance-based SCAFFOLD strength (capped)
        4. Layer-wise correction scaling (gentler: 0.85x-1.15x)
        5. DECOUPLED SCAFFOLD (v6 critical fix)
        """
        if self.client_control is None or self.server_control is None:
            self._init_control_variates()

        self.model.train()

        if config is None:
            config = {}
        server_round = int(config.get("server_round", 1))
        num_rounds = int(config.get("num_rounds", 40))
        warmup_rounds = int(config.get("warmup_rounds", 3))

        # Disable warm-start by default - it was causing instability
        use_warm_start = config.get("use_warm_start", False)
        if server_round == 1 and use_warm_start:
            self._warm_start_control_variates()

        # Track staleness for clients that skip rounds
        self.control_staleness = server_round - self.last_participated_round - 1
        staleness_factor = 1.0 / (1.0 + 0.1 * max(0, self.control_staleness))

        # ================================================================
        # IMPROVEMENT 1: Three-Phase Scheduling
        # ================================================================
        exploration_end = warmup_rounds + int((num_rounds - warmup_rounds) * 0.3)
        transition_end = warmup_rounds + int((num_rounds - warmup_rounds) * 0.7)

        if server_round <= warmup_rounds:
            # WARMUP: Very gentle start - minimal SCAFFOLD
            warmup_progress = server_round / max(warmup_rounds, 1)
            base_scaffold = 0.2 * warmup_progress
            base_fedprox = 0.05 * warmup_progress
            phase = "warmup"

        elif server_round <= exploration_end:
            # EXPLORATION: SCAFFOLD dominant but not too aggressive
            base_scaffold = 0.8
            base_fedprox = 0.15
            phase = "exploration"

        elif server_round <= transition_end:
            # TRANSITION: Gradual shift
            progress = (server_round - exploration_end) / max(transition_end - exploration_end, 1)
            base_scaffold = 0.8 - 0.2 * progress  # 0.8 → 0.6
            base_fedprox = 0.15 + 0.35 * progress  # 0.15 → 0.5
            phase = "transition"

        else:
            # CONVERGENCE: FedProx increases for stability
            progress = (server_round - transition_end) / max(num_rounds - transition_end, 1)
            base_scaffold = 0.6 - 0.1 * progress  # 0.6 → 0.5
            base_fedprox = 0.5 + 0.3 * progress   # 0.5 → 0.8
            phase = "convergence"

        # ================================================================
        # IMPROVEMENT 2: Per-Client Adaptive μ
        # ================================================================
        client_divergence = self._compute_client_divergence()

        baseline_divergence = config.get("baseline_divergence", 1.0)
        divergence_ratio = client_divergence / baseline_divergence

        # Conservative scaling range (0.7-1.5)
        divergence_scale = np.clip(divergence_ratio, 0.7, 1.5)

        adaptive_mu = base_fedprox * self.mu * self.fedprox_weight * divergence_scale

        # ================================================================
        # Setup optimizer and scheduler
        # ================================================================
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.trainloader) * self.local_epochs,
            eta_min=self.learning_rate * 0.1
        )

        initial_params = [p.clone().detach() for p in self.model.parameters()]
        num_params = len(initial_params)

        # v6: Precompute SCAFFOLD corrections for efficiency
        scaffold_correction = []
        for c_i, c in zip(self.client_control, self.server_control):
            correction = c.to(self.device) - c_i.to(self.device)
            scaffold_correction.append(correction)

        total_loss = 0.0
        n_batches = 0
        total_steps = 0
        batch_gradient_norms = []

        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                mse_loss = self.criterion(y_pred, y_batch)

                # FedProx proximal term in loss (this stays in the loss)
                proximal_term = 0.0
                if self.global_params is not None and adaptive_mu > 0:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p.to(self.device)) ** 2)
                    proximal_term = (adaptive_mu / 2.0) * proximal_term

                loss = mse_loss + proximal_term

                optimizer.zero_grad()
                loss.backward()

                # ================================================================
                # IMPROVEMENT 3: Gradient Variance-Based SCAFFOLD Strength
                # ================================================================
                grad_norm = sum(
                    p.grad.norm().item() ** 2 for p in self.model.parameters()
                    if p.grad is not None
                ) ** 0.5
                batch_gradient_norms.append(grad_norm)

                # Compute variance factor from recent gradients
                if len(batch_gradient_norms) >= 5:
                    recent_norms = batch_gradient_norms[-10:]
                    grad_mean = np.mean(recent_norms)
                    grad_variance = np.var(recent_norms)

                    # Tighter cap (1.2x) and smaller scaling
                    variance_factor = min(1.2, 1.0 + 0.5 * grad_variance / (grad_mean + 1e-8))
                else:
                    variance_factor = 1.0

                # Clip gradients BEFORE optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                # ================================================================
                # v6 KEY FIX: Optimizer step with UNMODIFIED gradients
                # This preserves Adam's momentum/variance estimates
                # ================================================================
                optimizer.step()
                scheduler.step()

                # Get current learning rate for proper scaling
                effective_lr = scheduler.get_last_lr()[0]

                # ================================================================
                # v6 KEY FIX: Apply SCAFFOLD as POST-OPTIMIZER parameter correction
                # This is the critical decoupling - SCAFFOLD doesn't interfere with Adam
                # ================================================================
                scaffold_strength = (
                    base_scaffold * self.scaffold_weight *
                    variance_factor * staleness_factor
                )

                # Apply SCAFFOLD correction to parameters (not gradients!)
                with torch.no_grad():
                    for param_idx, (param, correction) in enumerate(zip(
                        self.model.parameters(),
                        scaffold_correction
                    )):
                        # Layer scaling: early layers less correction, later layers more
                        layer_progress = param_idx / max(num_params - 1, 1)

                        # Gentler scaling (0.85x-1.15x)
                        layer_scaffold_scale = 0.85 + 0.3 * layer_progress

                        layer_scaffold = scaffold_strength * layer_scaffold_scale

                        # v6: Scale by learning rate for proper magnitude
                        # Apply as parameter update, not gradient modification
                        param.add_(layer_scaffold * effective_lr * correction)

                total_loss += mse_loss.item()
                n_batches += 1
                total_steps += 1

        # Update gradient norm history (keep last 100)
        self.gradient_norm_history.extend(batch_gradient_norms)
        self.gradient_norm_history = self.gradient_norm_history[-100:]

        # ================================================================
        # Control Variate Update
        # ================================================================
        old_client_control = [c.clone() for c in self.client_control]

        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control,
                initial_params, self.model.parameters()
            )):
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / total_steps
                self.client_control[i] = c_i.to(self.device) - c.to(self.device) + param_diff

        delta_control = [
            (new_c - old_c).cpu().numpy()
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]

        # Update participation tracking
        self.last_participated_round = server_round

        return total_loss / n_batches if n_batches > 0 else 0.0, delta_control
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        
        self.local_epochs = int(config.get("local_epochs", self.local_epochs))
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.mu = float(config.get("mu", self.mu))
        self.scaffold_lr_correction = float(config.get("scaffold_lr_correction", self.scaffold_lr_correction))
        self.fedprox_weight = float(config.get("fedprox_weight", self.fedprox_weight))
        self.scaffold_weight = float(config.get("scaffold_weight", self.scaffold_weight))
        self.max_grad_norm = float(config.get("max_grad_norm", self.max_grad_norm))
        
        if "server_control" in config and config["server_control"] is not None:
            server_control_list = config["server_control"]
            self.server_control = [
                torch.tensor(np.array(c), dtype=torch.float32, device=self.device)
                for c in server_control_list
            ]
            if self.client_control is None:
                self._init_control_variates()
        
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
        
        new_parameters = self.get_parameters({})
        num_samples = len(self.trainloader.dataset)
        
        metrics = {
            "client_id": self.cid,
            "train_loss": float(loss),
            "num_samples": num_samples,
            "strategy": self.strategy
        }
        
        if delta_control is not None:
            metrics["delta_control"] = [dc.tolist() if hasattr(dc, 'tolist') else dc for dc in delta_control]
        
        # v3: Add client divergence to metrics for monitoring
        if self.strategy == "hybrid":
            metrics["client_divergence"] = self._compute_client_divergence()
        
        return new_parameters, num_samples, metrics
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
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
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        num_samples = len(all_targets)
        
        avg_loss = total_loss / num_samples
        
        preds_original = all_preds * self.target_std + self.target_mean
        targets_original = all_targets * self.target_std + self.target_mean
        
        if self.log_transform:
            preds_original = torch.expm1(preds_original)
            targets_original = torch.expm1(targets_original)
        
        mse = torch.mean((preds_original - targets_original) ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(preds_original - targets_original)).item()
        
        ss_res = torch.sum((targets_original - preds_original) ** 2).item()
        ss_tot = torch.sum((targets_original - torch.mean(targets_original)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
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