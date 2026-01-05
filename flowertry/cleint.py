from collections import OrderedDict
from typing import Dict, Optional
import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import Net, train, test


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
        
        # Check if this is FedSCAFFOLD
        is_scaffold = 'scaffold_lr' in config
        c_global = config.get('c_global', None)
        
        if is_scaffold:
            # Initialize or update client control variate
            if self.c_client is None:
                self.c_client = [np.zeros_like(p) for p in parameters]
            elif 'c_client' in config:
                self.c_client = config['c_client']
            
            # Train with SCAFFOLD correction
            scaffold_lr = config.get('scaffold_lr', 1.0)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            self._train_scaffold(optimizer, epochs, c_global, scaffold_lr)
            
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
            # Regular training (FedAvg/FedProx)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            train(self.model, self.trainloader, optimizer, epochs, self.device)
            
            return self.get_parameters({}), len(self.trainloader.dataset), {}
    
    def _train_scaffold(self, optimizer, epochs, c_global, scaffold_lr):
        """Train with SCAFFOLD correction term."""
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
                optimizer.step()

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
