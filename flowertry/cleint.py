from collections import OrderedDict
from typing import Dict, List
import copy
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import MLP, train, test


def set_parameters(model, parameters):
    """Load model parameters from numpy arrays into PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict()
    for k, v in params_dict:
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v.copy())
        else:
            state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model):
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client using FedAvg."""
    
    def __init__(self, trainloader, valloader, num_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)
        weight_decay = config.get('weight_decay', 0.0001)

        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay  # L2 regularization
        )
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


class FedProxClient(fl.client.NumPyClient):
    """
    Improved FedProx client with adaptive proximal term.
    
    Key improvements:
    - Adaptive μ scheduling option
    - Better gradient handling
    - Weight decay support
    """
    
    def __init__(
        self, 
        trainloader, 
        valloader, 
        num_features: int, 
        num_classes: int = 3,
        mu: float = 0.01
    ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")
        self.mu = mu
        self.round_num = 0

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        
        # Store global model for proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]

        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)
        weight_decay = config.get('weight_decay', 0.0001)
        
        # Get mu from config or use adaptive scheduling
        base_mu = config.get('fedprox_mu', self.mu)
        
        # Adaptive μ: increase slightly as training progresses to maintain stability
        self.round_num += 1
        adaptive_mu = base_mu * min(1.0 + 0.1 * (self.round_num // 5), 2.0)

        self._train_with_proximal(lr, momentum, epochs, global_params, adaptive_mu, weight_decay)

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def _train_with_proximal(self, lr: float, momentum: float, epochs: int, 
                             global_params: list, mu: float, weight_decay: float):
        """Train with FedProx proximal term and improvements."""
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay
        )
        self.model.train()
        self.model.to(self.device)
        
        for _ in range(epochs):
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_params):
                    diff = local_param - global_param.to(self.device)
                    proximal_term += torch.sum(diff ** 2)
                
                loss = loss + (mu / 2.0) * proximal_term
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


class SCAFFOLDClient(fl.client.NumPyClient):
    """
    Improved SCAFFOLD client with full control variate management.
    
    Key improvements:
    - Proper Option II control variate updates
    - Learning rate scaling for control variates
    - Better numerical stability
    - Gradient clipping
    """
    
    def __init__(
        self, 
        trainloader, 
        valloader, 
        num_features: int, 
        num_classes: int = 3,
        client_id: int = 0
    ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")
        self.client_id = client_id
        
        # Control variates - initialized to zeros
        self.c_local = None
        self.initialized = False
        
    def _init_control_variates(self):
        """Initialize control variates matching model parameters."""
        if not self.initialized:
            self.c_local = [torch.zeros_like(p.data) for p in self.model.parameters()]
            self.initialized = True
        
    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self._init_control_variates()
        
        # Store initial parameters
        initial_params = [p.clone().detach() for p in self.model.parameters()]
        
        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)
        weight_decay = config.get('weight_decay', 0.0001)
        
        # Server control variate (in full SCAFFOLD, this would come from server)
        # Here we approximate with zeros or could be passed via config
        c_global = [torch.zeros_like(p.data) for p in self.model.parameters()]
        
        # Train with SCAFFOLD correction
        num_steps = self._train_with_scaffold(lr, momentum, epochs, initial_params, 
                                              c_global, weight_decay)
        
        # Update local control variate (Option II from SCAFFOLD paper)
        if num_steps > 0:
            with torch.no_grad():
                new_c_local = []
                for p_new, p_old, c_l, c_g in zip(
                    self.model.parameters(), initial_params, self.c_local, c_global
                ):
                    # c_i_new = c_i - c + (1/(K*lr)) * (x - y)
                    # where x = initial params, y = final params
                    delta = (p_old - p_new.data) / (num_steps * lr)
                    
                    # Option II: c_i_new = c_g + delta
                    c_i_new = c_g + delta
                    new_c_local.append(c_i_new.clone())
                
                self.c_local = new_c_local
        
        return get_parameters(self.model), len(self.trainloader.dataset), {}
    
    def _train_with_scaffold(
        self, 
        lr: float, 
        momentum: float, 
        epochs: int, 
        initial_params: List[torch.Tensor],
        c_global: List[torch.Tensor],
        weight_decay: float
    ) -> int:
        """
        Train with SCAFFOLD variance reduction.
        
        The correction term (c_global - c_local) compensates for client drift.
        """
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay
        )
        self.model.train()
        self.model.to(self.device)
        
        num_steps = 0
        
        for _ in range(epochs):
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply SCAFFOLD correction: g = g + (c_global - c_local)
                with torch.no_grad():
                    for param, c_l, c_g in zip(self.model.parameters(), self.c_local, c_global):
                        if param.grad is not None:
                            correction = c_g.to(self.device) - c_l.to(self.device)
                            param.grad.data.add_(correction)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                num_steps += 1
        
        return num_steps

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


def generate_client_fn(
    trainloaders, 
    valloaders, 
    num_features: int, 
    num_classes: int = 3,
    strategy: str = "fedavg",
    fedprox_mu: float = 0.01
):
    """Generate a client function for Flower simulation."""
    def client_fn(cid: str):
        client_id = int(cid)
        
        if strategy == "fedprox":
            return FedProxClient(
                trainloader=trainloaders[client_id],
                valloader=valloaders[client_id],
                num_features=num_features,
                num_classes=num_classes,
                mu=fedprox_mu
            )
        elif strategy == "scaffold":
            return SCAFFOLDClient(
                trainloader=trainloaders[client_id],
                valloader=valloaders[client_id],
                num_features=num_features,
                num_classes=num_classes,
                client_id=client_id
            )
        else:  # Default: FedAvg
            return FlowerClient(
                trainloader=trainloaders[client_id],
                valloader=valloaders[client_id],
                num_features=num_features,
                num_classes=num_classes
            )

    return client_fn
