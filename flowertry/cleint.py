from collections import OrderedDict
from typing import Dict, Optional
import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import Net, train, train_fedprox, train_scaffold, train_hybrid_fedprox_scaffold, test, DEFAULT_INPUT_DIM


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for Savings Potential Classification.
    
    Uses MLP model for 3-class classification:
    - Low savings
    - Medium savings
    - High savings
    """
    
    def __init__(self, trainloader, valloader, num_classes: int = 3, class_weights=None, input_dim: int = DEFAULT_INPUT_DIM) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes, input_dim=input_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights
        self.input_dim = input_dim
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
        max_grad_norm = config.get('max_grad_norm', 1.0)

        if config.get("scaffold", False):
            c_global = config["c_global"]
            c_i = config["c_local"]

            # Defensive init (first participation)
            if self.c_client is None:
                self.c_client = [ci.copy() for ci in c_i]

            # Snapshot before training
            c_i_old = [ci.copy() for ci in self.c_client]

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum
            )

            # Perform SCAFFOLD local training
            c_i_new = train_scaffold(
                model=self.model,
                trainloader=self.trainloader,
                optimizer=optimizer,
                epochs=epochs,
                device=self.device,
                c_global=c_global,
                c_i=self.c_client,
                max_grad_norm=max_grad_norm,
                class_weights=self.class_weights
            )

            # Compute delta_c explicitly (REQUIRED)
            delta_c = [
                c_i_new[j] - c_i_old[j]
                for j in range(len(c_i_old))
            ]

            # Update local cache ONLY (server owns truth)
            self.c_client = c_i_new

            metrics = {
                "delta_c": delta_c
            }

            return self.get_parameters({}), len(self.trainloader.dataset), metrics

        elif config.get("hybrid", False):
            # Enhanced Hybrid FedProx-SCAFFOLD training
            # Supports: Sequential Activation, Conditional Activation, Dual-μ Architecture
            c_global = config["c_global"]
            c_i = config["c_local"]
            proximal_mu = config.get("proximal_mu", 0.0)
            use_scaffold = config.get("use_scaffold", True)
            use_dual_mu = config.get("use_dual_mu", False)
            mu_raw = config.get("mu_raw", 0.1)
            mu_corrected = config.get("mu_corrected", 0.001)
            activation_mode = config.get("activation_mode", "hybrid")

            # Defensive init (first participation)
            if self.c_client is None:
                self.c_client = [ci.copy() for ci in c_i]

            # Snapshot before training
            c_i_old = [ci.copy() for ci in self.c_client]

            # Store global parameters for FedProx proximal term
            global_params = [p.copy() for p in parameters]

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum
            )

            # Perform Enhanced Hybrid FedProx-SCAFFOLD local training
            c_i_new = train_hybrid_fedprox_scaffold(
                model=self.model,
                trainloader=self.trainloader,
                optimizer=optimizer,
                epochs=epochs,
                device=self.device,
                global_params=global_params,
                proximal_mu=proximal_mu,
                c_global=c_global,
                c_i=self.c_client,
                max_grad_norm=max_grad_norm,
                class_weights=self.class_weights,
                use_scaffold=use_scaffold,
                use_dual_mu=use_dual_mu,
                mu_raw=mu_raw,
                mu_corrected=mu_corrected
            )

            # Compute delta_c explicitly (REQUIRED for SCAFFOLD component)
            delta_c = [
                c_i_new[j] - c_i_old[j]
                for j in range(len(c_i_old))
            ]

            # Update local cache
            self.c_client = c_i_new

            metrics = {
                "delta_c": delta_c,
                "proximal_mu": proximal_mu,
                "use_dual_mu": int(use_dual_mu),
                "activation_mode": activation_mode
            }

            return self.get_parameters({}), len(self.trainloader.dataset), metrics

        else:
            # Check if this is FedProx (has proximal_mu in config)
            proximal_mu = config.get('proximal_mu', None)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

            if proximal_mu is not None and proximal_mu > 0:
                # FedProx training with proximal term
                # Store global parameters before training
                global_params = [p.copy() for p in parameters]
                train_fedprox(
                    self.model, self.trainloader, optimizer, epochs, self.device,
                    global_params=global_params,
                    proximal_mu=proximal_mu,
                    max_grad_norm=max_grad_norm,
                    class_weights=self.class_weights
                )
            else:
                # Regular FedAvg training
                train(self.model, self.trainloader, optimizer, epochs, self.device, max_grad_norm, self.class_weights)

            # Prepare metrics - include heterogeneity metrics if requested
            metrics = {}
            if config.get('report_heterogeneity', False):
                metrics.update(self.get_heterogeneity_metrics())

            return self.get_parameters({}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device, self.class_weights)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


class ScaffoldFlowerClient(FlowerClient):
    """
    SCAFFOLD-aware Flower client that retrieves control variates from strategy.

    This client extends FlowerClient to support SCAFFOLD by maintaining
    a reference to the strategy instance to retrieve control variates.
    """

    def __init__(self, trainloader, valloader, num_classes: int = 3, strategy=None, class_weights=None, input_dim: int = DEFAULT_INPUT_DIM):
        super().__init__(trainloader, valloader, num_classes, class_weights, input_dim)
        self.strategy = strategy

    def fit(self, parameters, config):
        """Train with SCAFFOLD using control variates from strategy."""
        # Check if SCAFFOLD and strategy is available
        if config.get('is_scaffold', False) and self.strategy is not None:
            client_id = config.get('client_id', '0')

            # Get control variates from strategy
            c_global, c_i = self.strategy.get_control_variates(client_id)

            # Add them to config for parent class to use
            if c_global is not None:
                config['c_global'] = c_global
            if c_i is not None:
                config['c_i'] = c_i

        # Call parent fit method
        return super().fit(parameters, config)


def generate_client_fn(trainloaders, valloaders, num_classes: int = 3, strategy=None, class_weights=None, input_dim: int = DEFAULT_INPUT_DIM):
    """
    Generate a client function for Flower simulation.

    Args:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        num_classes: Number of output classes (3 for savings classification)
        strategy: Optional strategy instance (for SCAFFOLD)
        class_weights: Optional class weights for weighted loss
        input_dim: Number of input features

    Returns:
        client_fn: Function that creates FlowerClient instances
    """
    def client_fn(cid: str):
        if strategy is not None:
            # Use SCAFFOLD-aware client
            return ScaffoldFlowerClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_classes=num_classes,
                strategy=strategy,
                class_weights=class_weights,
                input_dim=input_dim
            )
        else:
            # Use standard client
            return FlowerClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_classes=num_classes,
                class_weights=class_weights,
                input_dim=input_dim
            )

    return client_fn
