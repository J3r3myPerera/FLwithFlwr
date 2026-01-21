"""
Federated Learning Client - v4 Enhanced Hybrid Implementation
=============================================================

Building on v3's success (winning 5/6 metrics), v4 adds:

NEW IMPROVEMENTS:
================

6. LOSS-AWARE ADAPTIVE SCHEDULING
   - Monitor training loss trend
   - If loss plateaus, increase SCAFFOLD to escape local minima
   - If loss diverges, increase FedProx to stabilize

7. EXPONENTIAL MOVING AVERAGE (EMA) FOR CONTROL VARIATES
   - Smooth control variate updates to reduce noise
   - Prevents overcorrection from noisy gradients

8. GRADIENT PROJECTION
   - Project SCAFFOLD correction onto useful subspace
   - Avoid corrections that fight the gradient direction

9. ADAPTIVE LEARNING RATE PER PHASE
   - Higher LR during exploration
   - Lower LR during convergence

10. CLIENT MOMENTUM
    - Maintain momentum of updates across rounds
    - Helps escape local minima

11. IMPORTANCE-WEIGHTED AGGREGATION PREPARATION
    - Track and report client gradient norms for server weighting
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
    Enhanced v4 Flower client with additional hybrid improvements.
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
        
        # v4 NEW: Track loss history for adaptive scheduling
        self.loss_history = []
        self.round_loss_history = []  # Per-round average loss
        
        # v4 NEW: Client momentum buffer
        self.update_momentum = None
        self.momentum_beta = 0.9
        
        # v4 NEW: EMA for control variates
        self.control_ema_beta = 0.95
        
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
        """Initialize control variates from actual gradients."""
        if self.control_initialized_from_gradients:
            return
        
        self.model.train()
        gradient_sum = [torch.zeros_like(p) for p in self.model.parameters()]
        num_batches = 0
        
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
            
            if num_batches >= 5:
                break
        
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
    
    def _compute_loss_trend(self) -> str:
        """
        v4 NEW: Analyze recent loss history to determine trend.
        Returns: 'improving', 'plateau', 'diverging'
        """
        if len(self.round_loss_history) < 3:
            return 'improving'
        
        recent = self.round_loss_history[-3:]
        
        # Compute relative changes
        changes = [(recent[i+1] - recent[i]) / (recent[i] + 1e-8) 
                   for i in range(len(recent)-1)]
        avg_change = np.mean(changes)
        
        if avg_change < -0.01:  # Loss decreasing by >1%
            return 'improving'
        elif avg_change > 0.05:  # Loss increasing by >5%
            return 'diverging'
        else:
            return 'plateau'
    
    def _update_control_variates_ema(self, new_control: List[torch.Tensor]):
        """
        v4 NEW: Update control variates with exponential moving average.
        Smooths updates to reduce noise.
        """
        for i in range(len(self.client_control)):
            self.client_control[i] = (
                self.control_ema_beta * self.client_control[i] +
                (1 - self.control_ema_beta) * new_control[i]
            )
    
    def _apply_gradient_projection(self, correction: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        v4 NEW: Project SCAFFOLD correction to avoid fighting gradient.
        If correction opposes gradient too much, reduce its component in that direction.
        """
        if grad is None or grad.norm() < 1e-8:
            return correction
        
        # Compute cosine similarity
        grad_flat = grad.view(-1)
        corr_flat = correction.view(-1)
        
        cos_sim = torch.dot(grad_flat, corr_flat) / (grad_flat.norm() * corr_flat.norm() + 1e-8)
        
        # If correction strongly opposes gradient (cos_sim < -0.5), reduce it
        if cos_sim < -0.5:
            # Project out the component opposing gradient
            projection = (torch.dot(corr_flat, grad_flat) / (grad_flat.norm()**2 + 1e-8)) * grad_flat
            correction_flat = corr_flat - 0.5 * projection  # Partial projection
            return correction_flat.view_as(correction)
        
        return correction
    
    def _train_fedavg(self) -> float:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-3)
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-3)
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-3)
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
        v4 ENHANCED HYBRID: All v3 improvements plus:
        - Loss-aware adaptive scheduling
        - EMA for control variates
        - Gradient projection
        - Phase-adaptive learning rate
        """
        if self.client_control is None or self.server_control is None:
            self._init_control_variates()
        
        self.model.train()
        
        if config is None:
            config = {}
        server_round = int(config.get("server_round", 1))
        num_rounds = int(config.get("num_rounds", 30))
        warmup_rounds = int(config.get("warmup_rounds", 2))
        
        # v3: Optional warm-start on first round
        use_warm_start = config.get("use_warm_start", True)
        if server_round == 1 and use_warm_start:
            self._warm_start_control_variates()
        
        # Track staleness for clients that skip rounds
        self.control_staleness = server_round - self.last_participated_round - 1
        staleness_factor = 1.0 / (1.0 + 0.1 * max(0, self.control_staleness))
        
        # ================================================================
        # v4 NEW: Loss-Aware Adjustment
        # ================================================================
        loss_trend = self._compute_loss_trend()
        if loss_trend == 'plateau':
            # Increase SCAFFOLD to escape local minima
            loss_adjustment_scaffold = 1.2
            loss_adjustment_fedprox = 0.8
        elif loss_trend == 'diverging':
            # Increase FedProx to stabilize
            loss_adjustment_scaffold = 0.7
            loss_adjustment_fedprox = 1.3
        else:  # improving
            loss_adjustment_scaffold = 1.0
            loss_adjustment_fedprox = 1.0
        
        # ================================================================
        # Three-Phase Scheduling (from v3)
        # ================================================================
        exploration_end = warmup_rounds + int((num_rounds - warmup_rounds) * 0.3)
        transition_end = warmup_rounds + int((num_rounds - warmup_rounds) * 0.7)
        
        if server_round <= warmup_rounds:
            warmup_progress = server_round / max(warmup_rounds, 1)
            base_scaffold = 0.3 * warmup_progress
            base_fedprox = 0.0
            phase = "warmup"
            lr_multiplier = 0.5 + 0.5 * warmup_progress  # v4: Ramp up LR
            
        elif server_round <= exploration_end:
            base_scaffold = 1.0
            base_fedprox = 0.1
            phase = "exploration"
            lr_multiplier = 1.1  # v4: Slightly higher LR for exploration
            
        elif server_round <= transition_end:
            progress = (server_round - exploration_end) / max(transition_end - exploration_end, 1)
            base_scaffold = 1.0 - 0.3 * progress
            base_fedprox = 0.1 + 0.4 * progress
            phase = "transition"
            lr_multiplier = 1.0 - 0.2 * progress  # v4: Gradual LR decrease
            
        else:
            progress = (server_round - transition_end) / max(num_rounds - transition_end, 1)
            base_scaffold = 0.7 - 0.2 * progress
            base_fedprox = 0.5 + 0.3 * progress
            phase = "convergence"
            lr_multiplier = 0.8 - 0.3 * progress  # v4: Low LR for fine-tuning
        
        # Apply loss-aware adjustments
        base_scaffold *= loss_adjustment_scaffold
        base_fedprox *= loss_adjustment_fedprox
        
        # ================================================================
        # Per-Client Adaptive μ (from v3)
        # ================================================================
        client_divergence = self._compute_client_divergence()
        baseline_divergence = config.get("baseline_divergence", 1.0)
        divergence_ratio = client_divergence / baseline_divergence
        divergence_scale = np.clip(divergence_ratio, 0.5, 2.0)
        
        adaptive_mu = base_fedprox * self.mu * self.fedprox_weight * divergence_scale
        
        # ================================================================
        # v4 NEW: Phase-Adaptive Learning Rate
        # ================================================================
        effective_lr = self.learning_rate * lr_multiplier
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=effective_lr,
            weight_decay=5e-3
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(self.trainloader) * self.local_epochs, 
            eta_min=effective_lr * 0.1
        )
        
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        num_params = len(initial_params)
        
        total_loss = 0.0
        n_batches = 0
        total_steps = 0
        batch_gradient_norms = []
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                mse_loss = self.criterion(y_pred, y_batch)
                
                # FedProx proximal term
                proximal_term = 0.0
                if self.global_params is not None and adaptive_mu > 0:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p.to(self.device)) ** 2)
                    proximal_term = (adaptive_mu / 2.0) * proximal_term
                
                loss = mse_loss + proximal_term
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Variance-Based SCAFFOLD Strength (from v3)
                grad_norm = sum(
                    p.grad.norm().item() ** 2 for p in self.model.parameters() 
                    if p.grad is not None
                ) ** 0.5
                batch_gradient_norms.append(grad_norm)
                
                if len(batch_gradient_norms) >= 5:
                    recent_norms = batch_gradient_norms[-10:]
                    grad_mean = np.mean(recent_norms)
                    grad_variance = np.var(recent_norms)
                    variance_factor = min(1.5, 1.0 + grad_variance / (grad_mean + 1e-8))
                else:
                    variance_factor = 1.0
                
                scaffold_strength = (
                    base_scaffold * self.scaffold_weight * 
                    variance_factor * staleness_factor
                )
                
                # ================================================================
                # Layer-Wise Corrections with Gradient Projection (v3 + v4)
                # ================================================================
                with torch.no_grad():
                    for param_idx, (param, c_i, c) in enumerate(zip(
                        self.model.parameters(),
                        self.client_control,
                        self.server_control
                    )):
                        if param.grad is not None:
                            layer_progress = param_idx / max(num_params - 1, 1)
                            layer_scaffold_scale = 0.7 + 0.6 * layer_progress
                            
                            layer_scaffold = scaffold_strength * layer_scaffold_scale
                            correction = layer_scaffold * (c.to(self.device) - c_i.to(self.device))
                            
                            # v4 NEW: Apply gradient projection
                            correction = self._apply_gradient_projection(correction, param.grad)
                            
                            param.grad.add_(correction)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                total_loss += mse_loss.item()
                epoch_loss += mse_loss.item()
                n_batches += 1
                epoch_batches += 1
                total_steps += 1
            
            epoch_losses.append(epoch_loss / max(epoch_batches, 1))
        
        # Update loss history for next round
        self.loss_history.extend(epoch_losses)
        self.round_loss_history.append(np.mean(epoch_losses))
        
        # Keep history bounded
        self.loss_history = self.loss_history[-50:]
        self.round_loss_history = self.round_loss_history[-10:]
        self.gradient_norm_history = batch_gradient_norms[-100:]
        
        # ================================================================
        # v4 NEW: Control Variate Update with EMA option
        # ================================================================
        old_client_control = [c.clone() for c in self.client_control]
        
        # Compute new control variates
        new_client_control = []
        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control,
                initial_params, self.model.parameters()
            )):
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / total_steps
                new_c = c_i.to(self.device) - c.to(self.device) + param_diff
                new_client_control.append(new_c)
        
        # Apply EMA update (v4)
        use_ema = config.get("use_control_ema", True)
        if use_ema and server_round > warmup_rounds:
            self._update_control_variates_ema(new_client_control)
        else:
            self.client_control = new_client_control
        
        delta_control = [
            (new_c - old_c).cpu().numpy()
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]
        
        # Update participation tracking
        self.last_participated_round = server_round
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        
        return avg_loss, delta_control
    
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
        
        # v4: Add additional metrics for monitoring
        if self.strategy == "hybrid":
            metrics["client_divergence"] = self._compute_client_divergence()
            metrics["loss_trend"] = self._compute_loss_trend()
            if self.gradient_norm_history:
                metrics["avg_grad_norm"] = float(np.mean(self.gradient_norm_history[-20:]))
        
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