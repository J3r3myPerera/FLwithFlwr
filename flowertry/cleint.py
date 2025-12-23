from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import Net, train, test


def get_device():
    """
    Get the best available device for training.
    
    Note: MPS (Apple Silicon GPU) is NOT used because it's incompatible with
    Ray's multiprocessing used by Flower simulation. MPS causes bus errors
    when used with process forking.
    
    For M1 Pro MacBook, CPU training is still efficient due to:
    - Unified memory architecture (no CPU-GPU data transfer overhead)
    - High single-core performance
    - Efficient NEON SIMD instructions
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        # Use CPU for Flower simulation (MPS incompatible with Ray multiprocessing)
        return torch.device("cpu")


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for FedAvg."""
    
    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        
        self.model = Net(num_classes)

        self.device = get_device()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        # Used for FedAvg algorithm
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}


class FedProxClient(fl.client.NumPyClient):
    """
    FedProx client that adds a proximal term to prevent client drift.
    
    The proximal term penalizes the local model from drifting too far
    from the global model: L_local = L_original + (mu/2) * ||w - w_global||^2
    
    This helps in non-IID settings where clients may have very different
    data distributions.
    """
    
    def __init__(self, trainloader, valloader, num_classes, mu: float = 0.1) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.mu = mu  # Proximal term coefficient
        
        self.model = Net(num_classes)
        self.device = get_device()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)
        
        # Store global model parameters for proximal term
        global_params = [val.clone().detach() for val in self.model.parameters()]

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        
        # Override mu if provided in config
        mu = config.get('mu', self.mu)

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training with proximal term
        self._train_with_proximal(global_params, optim, epochs, mu)

        return self.get_parameters({}), len(self.trainloader), {}

    def _train_with_proximal(self, global_params, optimizer, epochs: int, mu: float):
        """
        Train the network with FedProx proximal term.
        
        Loss = CrossEntropy + (mu/2) * ||w - w_global||^2
        """
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        self.model.to(self.device)
        
        # Move global params to device
        global_params = [p.to(self.device) for p in global_params]
        
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Standard cross-entropy loss
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Add proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_params):
                    proximal_term += ((local_param - global_param) ** 2).sum()
                
                loss += (mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes, strategy: str = "fedavg", mu: float = 0.1):
    """
    Generate client function based on the selected strategy.
    
    Args:
        trainloaders: List of training data loaders
        valloaders: List of validation data loaders
        num_classes: Number of output classes
        strategy: "fedavg" or "fedprox"
        mu: Proximal term coefficient (only used for FedProx)
    
    Returns:
        client_fn: Function that creates clients by ID
    """
    def client_fn(cid: str):
        if strategy == "fedprox":
            return FedProxClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_classes=num_classes,
                mu=mu
            )
        else:  # Default to FedAvg
            return FlowerClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_classes=num_classes
            )

    return client_fn