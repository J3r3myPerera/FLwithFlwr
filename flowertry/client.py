from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import Net, test


def get_device():
    """
    Get the best available device for Flower simulation.
    
    Note: MPS (Apple Silicon) is disabled because Ray's multiprocessing
    (used by Flower simulation) causes segmentation faults with MPS.
    For single-process training, MPS works fine.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        # Use CPU for Flower simulation (MPS + Ray = segfault)
        return torch.device("cpu")


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client implementation with FedProx support for non-IID data.
    Optimized for Apple Silicon (M1/M2) with MPS backend.
    """
    
    def __init__(self, trainloader, valloader, num_classes, 
                 use_fedprox: bool = False, fedprox_mu: float = 0.01) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes)
        self.device = get_device()
        self.use_fedprox = use_fedprox
        self.fedprox_mu = fedprox_mu

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model on local data."""
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)
        
        # Get global model parameters for FedProx proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # Extract training configuration
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        use_fedprox = config.get('use_fedprox', self.use_fedprox)
        fedprox_mu = config.get('fedprox_mu', self.fedprox_mu)

        # Create optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Perform local training
        if use_fedprox:
            self._train_fedprox(optimizer, epochs, global_params, fedprox_mu)
        else:
            self._train_fedavg(optimizer, epochs)

        # Return updated model parameters
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def _train_fedavg(self, optimizer, epochs):
        """Standard FedAvg training."""
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        self.model.to(self.device)
        
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

    def _train_fedprox(self, optimizer, epochs, global_params, mu):
        """
        FedProx training with proximal regularization.
        
        The proximal term penalizes the distance between local and global model,
        which helps with non-IID data by preventing clients from drifting too far
        from the global model.
        
        Loss = CrossEntropy + (mu/2) * ||w - w_global||^2
        """
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        self.model.to(self.device)
        
        # Move global parameters to device
        global_params = [p.to(self.device) for p in global_params]
        
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                # Standard cross-entropy loss
                output = self.model(images)
                loss = criterion(output, labels)
                
                # Add proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_params):
                    proximal_term += torch.sum((local_param - global_param) ** 2)
                
                loss += (mu / 2.0) * proximal_term
                
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on local validation data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


def generate_client_fn(trainloaders, valloaders, num_classes, 
                       use_fedprox: bool = False, fedprox_mu: float = 0.01):
    """
    Generate the client function for Flower simulation.
    
    Args:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        num_classes: Number of output classes
        use_fedprox: Whether to use FedProx regularization
        fedprox_mu: FedProx proximal term coefficient
    
    Returns:
        Client function that creates FlowerClient instances
    """
    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
            use_fedprox=use_fedprox,
            fedprox_mu=fedprox_mu
        )

    return client_fn

