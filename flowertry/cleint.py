from collections import OrderedDict
from typing import Dict, Optional
import random
import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import Net, train, train_fedprox, train_fedprox_scaffold, test


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for Savings Potential Classification.
    
    Uses MLP model for 3-class classification:
    - Low (<7% savings)
    - Medium (7-12% savings)
    - High (>12% savings)
    """
    
    def __init__(self, trainloader, valloader, num_classes: int = 3) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Control variate for FedSCAFFOLD
        self.c_client: Optional[NDArrays] = None
        # Cache heterogeneity metrics (computed once per client)
        self._label_entropy: Optional[float] = None
        self._feature_variance: Optional[float] = None
    
    def _compute_label_entropy(self) -> float:
        """Compute label distribution entropy (higher = more uniform, lower = more skewed)."""
        if self._label_entropy is not None:
            return self._label_entropy
        
        all_labels = []
        for _, labels in self.trainloader:
            all_labels.extend(labels.numpy().tolist())
        
        labels_arr = np.array(all_labels)
        _, counts = np.unique(labels_arr, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        
        # Return skewness score (1 - normalized entropy): higher = more skewed
        self._label_entropy = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return self._label_entropy
    
    def _compute_feature_variance(self) -> float:
        """Compute mean feature variance across all features."""
        if self._feature_variance is not None:
            return self._feature_variance
        
        all_features = []
        for features, _ in self.trainloader:
            all_features.append(features.numpy())
        
        features_arr = np.vstack(all_features)
        self._feature_variance = float(np.mean(np.var(features_arr, axis=0)))
        return self._feature_variance
    
    def get_heterogeneity_metrics(self) -> Dict[str, float]:
        """Get local heterogeneity metrics for adaptive mu computation."""
        return {
            'label_entropy': self._compute_label_entropy(),
            'feature_variance': self._compute_feature_variance()
        }

    def set_parameters(self, parameters):
        """Load model parameters from server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters to send to server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model on local data."""
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        
        # Check if this is hybrid FedProx-SCAFFOLD
        is_hybrid = config.get('is_hybrid', False)
        # Check if this is FedSCAFFOLD only
        is_scaffold = 'scaffold_lr' in config and not is_hybrid
        c_global = config.get('c_global', None)
        
        if is_hybrid:
            # Hybrid FedProx-SCAFFOLD training
            if random.random() < 0.01:  # 1% chance to print
                print(f"  [Hybrid] Training with proximal_mu={config.get('proximal_mu', 0):.4f}, scaffold_lr={config.get('scaffold_lr', 0):.4f}")
            return self._fit_hybrid(parameters, config)
        elif is_scaffold:
            # Initialize or update client control variate
            if self.c_client is None:
                self.c_client = [np.zeros_like(p) for p in parameters]
            elif 'c_client' in config:
                self.c_client = config['c_client']
            
            # Train with SCAFFOLD correction
            scaffold_lr = config.get('scaffold_lr', 1.0)
            max_grad_norm = config.get('max_grad_norm', 1.0)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            self._train_scaffold(optimizer, epochs, c_global, scaffold_lr, max_grad_norm)
            
            # Compute control variate update
            params_after = self.get_parameters({})
            params_before = [p.copy() for p in parameters]
            
            # c_update = (params_after - params_before) / (lr * epochs)
            # Simplified version: difference in parameters
            c_update = [
                (pa - pb) / (lr * epochs) if lr * epochs > 0 else (pa - pb)
                for pa, pb in zip(params_after, params_before)
            ]
            
            # Update client control variate
            if c_global is not None:
                c_client_new = [
                    cc + (cg - cc) / (len(self.trainloader.dataset) * epochs)
                    if len(self.trainloader.dataset) * epochs > 0 else cc
                    for cc, cg in zip(self.c_client, c_global)
                ]
            else:
                c_client_new = self.c_client
            
            metrics = {
                'c_update': c_update,
                'c_client_new': c_client_new
            }
            
            return params_after, len(self.trainloader.dataset), metrics
        else:
            # Check if this is FedProx (has proximal_mu in config)
            proximal_mu = config.get('proximal_mu', None)
            max_grad_norm = config.get('max_grad_norm', 1.0)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            
            if proximal_mu is not None and proximal_mu > 0:
                # FedProx training with proximal term
                # Store global parameters before training
                global_params = [p.copy() for p in parameters]
                # Debug: Print that FedProx is being used (only occasionally)
                if random.random() < 0.01:  # 1% chance to print
                    print(f"  [FedProx] Training with proximal_mu={proximal_mu:.4f}")
                train_fedprox(
                    self.model, self.trainloader, optimizer, epochs, self.device,
                    global_params=global_params,
                    proximal_mu=proximal_mu,
                    max_grad_norm=max_grad_norm
                )
            else:
                # Regular FedAvg training
                train(self.model, self.trainloader, optimizer, epochs, self.device, max_grad_norm)
            
            # Prepare metrics - include heterogeneity metrics if requested
            metrics = {}
            if config.get('report_heterogeneity', False):
                metrics.update(self.get_heterogeneity_metrics())
            
            return self.get_parameters({}), len(self.trainloader.dataset), metrics
    
    def _train_scaffold(self, optimizer, epochs, c_global, scaffold_lr, max_grad_norm: float = 1.0):
        """Train with SCAFFOLD correction term.
        
        Args:
            optimizer: Optimizer for training
            epochs: Number of training epochs
            c_global: Global control variate
            scaffold_lr: SCAFFOLD learning rate
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        import torch.nn.functional as F
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        self.model.to(self.device)
        
        # Convert control variates to tensors
        c_global_tensors = None
        c_client_tensors = None
        if c_global is not None:
            param_dict = dict(self.model.named_parameters())
            c_global_tensors = {
                name: torch.tensor(cg, device=self.device, requires_grad=False)
                for (name, _), cg in zip(param_dict.items(), c_global)
            }
        if self.c_client is not None:
            param_dict = dict(self.model.named_parameters())
            c_client_tensors = {
                name: torch.tensor(cc, device=self.device, requires_grad=False)
                for (name, _), cc in zip(param_dict.items(), self.c_client)
            }
        
        for _ in range(epochs):
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Add SCAFFOLD correction term
                if c_global_tensors is not None and c_client_tensors is not None:
                    scaffold_correction = 0.0
                    for name, param in self.model.named_parameters():
                        if name in c_global_tensors and name in c_client_tensors:
                            correction = scaffold_lr * (c_global_tensors[name] - c_client_tensors[name])
                            scaffold_correction += (correction * param).sum()
                    loss = loss - scaffold_correction
                
                loss.backward()
                # Clip gradients to prevent exploding gradients
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

    def _fit_hybrid(self, parameters, config):
        """
        Train with hybrid FedProx-SCAFFOLD approach.
        
        Combines proximal regularization with SCAFFOLD control variates
        for improved convergence on heterogeneous data.
        """
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # FedProx parameters
        proximal_mu = config.get('proximal_mu', 0.1)
        
        # SCAFFOLD parameters
        scaffold_lr = config.get('scaffold_lr', 1.0)
        c_global = config.get('c_global', None)
        
        # Initialize or update client control variate
        if self.c_client is None:
            self.c_client = [np.zeros_like(p) for p in parameters]
        elif 'c_client' in config:
            self.c_client = config['c_client']
        
        # Store global parameters before training
        global_params = [p.copy() for p in parameters]
        
        # Create optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        # Train with hybrid approach
        train_fedprox_scaffold(
            self.model, self.trainloader, optimizer, epochs, self.device,
            global_params=global_params,
            proximal_mu=proximal_mu,
            c_global=c_global,
            c_client=self.c_client,
            scaffold_lr=scaffold_lr,
            max_grad_norm=max_grad_norm
        )
        
        # Get updated parameters
        params_after = self.get_parameters({})
        
        # Compute control variate update (SCAFFOLD-style)
        c_update = [
            (pa - pb) / (lr * epochs) if lr * epochs > 0 else (pa - pb)
            for pa, pb in zip(params_after, global_params)
        ]
        
        # Update client control variate
        if c_global is not None:
            num_samples = len(self.trainloader.dataset)
            c_client_new = [
                cc + (cu - cg) * (num_samples / (num_samples * epochs + 1))
                for cc, cu, cg in zip(self.c_client, c_update, c_global)
            ]
        else:
            c_client_new = self.c_client
        
        # Prepare metrics
        metrics = {
            'c_update': c_update,
            'c_client_new': c_client_new,
            'proximal_mu': proximal_mu,
            'scaffold_lr': scaffold_lr,
            'prox_weight': config.get('prox_weight', 0.5),
            'scaffold_weight': config.get('scaffold_weight', 0.5)
        }
        
        # Include heterogeneity metrics if requested
        if config.get('report_heterogeneity', False):
            metrics.update(self.get_heterogeneity_metrics())
        
        return params_after, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes: int = 3):
    """
    Generate a client function for Flower simulation.
    
    Args:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        num_classes: Number of output classes (3 for savings classification)
    
    Returns:
        client_fn: Function that creates FlowerClient instances
    """
    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes
        )

    return client_fn
