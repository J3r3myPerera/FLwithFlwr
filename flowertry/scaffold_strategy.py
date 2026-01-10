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
import torch


class FedScaffoldStrategy(fl.server.strategy.Strategy):
    """
    Correct SCAFFOLD (Option II) implementation.

    Paper:
    Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    ICML 2020
    """

    def __init__(
        self,
        min_fit_clients: int,
        min_available_clients: int,
        min_evaluate_clients: int,
        on_fit_config_fn=None,
        evaluate_fn=None,
        total_clients: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        test_loader=None,
        criterion=None,
        config: Optional[dict] = None,
    ):
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.min_evaluate_clients = min_evaluate_clients

        self.on_fit_config_fn = on_fit_config_fn
        self.user_evaluate_fn = evaluate_fn  # user-provided fn

        if total_clients is None:
            raise ValueError("total_clients must be provided for SCAFFOLD")
        self.total_clients = total_clients

        self.current_parameters: Optional[List] = None
        self.c_global: Optional[List[np.ndarray]] = None
        self.client_control_variates: Dict[str, List[np.ndarray]] = {}

        # PyTorch evaluation setup
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config or {}

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        return None

    # -------------------------------------------------------------------------
    # Client Sampling
    # -------------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        if self.current_parameters is None:
            self.current_parameters = parameters_to_ndarrays(parameters)

        # Initialize global control variate c_global if not done
        if self.c_global is None:
            self.c_global = [np.zeros_like(p) for p in self.current_parameters]

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

            # Initialize per-client control variate c_i
            if cid not in self.client_control_variates:
                self.client_control_variates[cid] = [
                    np.zeros_like(p) for p in self.current_parameters
                ]

            config_dict = {}
            if self.on_fit_config_fn is not None:
                config_dict.update(self.on_fit_config_fn(server_round))

            # Inject SCAFFOLD info
            config_dict["scaffold"] = True
            config_dict["c_global"] = self.c_global
            config_dict["c_local"] = self.client_control_variates[cid]

            fit_ins = FitIns(parameters, config_dict)
            fit_instructions.append((client, fit_ins))

        return fit_instructions

    # -------------------------------------------------------------------------
    # Aggregation (SCAFFOLD logic)
    # -------------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        deltas_w = []
        delta_cs = []
        num_examples = []

        for client, fit_res in results:
            cid = client.cid
            if fit_res.num_examples == 0:
                continue

            # Model delta
            w_i = parameters_to_ndarrays(fit_res.parameters)
            delta_w = [w_i[j] - self.current_parameters[j] for j in range(len(w_i))]
            deltas_w.append(delta_w)
            num_examples.append(fit_res.num_examples)

            # Control variate delta
            if "delta_c" not in fit_res.metrics:
                raise RuntimeError(f"Client {cid} did not return delta_c")

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

        # Aggregate model updates (weighted)
        total_examples = sum(num_examples)
        avg_delta_w = [
            sum(deltas_w[i][j] * num_examples[i] for i in range(len(deltas_w)))
            / total_examples
            for j in range(len(deltas_w[0]))
        ]
        self.current_parameters = [
            self.current_parameters[j] + avg_delta_w[j] for j in range(len(avg_delta_w))
        ]

        # Aggregate control variates (Option II)
        avg_delta_c = [
            sum(delta_cs[i][j] for i in range(len(delta_cs))) / self.total_clients
            for j in range(len(self.c_global))
        ]
        self.c_global = [
            self.c_global[j] + avg_delta_c[j] for j in range(len(self.c_global))
        ]

        metrics = {
            "c_global_norm": np.sqrt(
                sum(np.sum(x * x) for x in self.c_global)
            ),
            "num_clients": len(results),
        }

        return ndarrays_to_parameters(self.current_parameters), metrics

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        if self.user_evaluate_fn is not None:
            return self.user_evaluate_fn(server_round, parameters, config=self.config)

        if self.model is None or self.test_loader is None or self.criterion is None:
            return None

        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in zip(self.model.state_dict().keys(), ndarrays)
        }
        self.model.load_state_dict(state_dict)

        # Evaluate on test set
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        batch_size = self.config.get("batch_size", None)

        with torch.no_grad():
            for X, y in self.test_loader:
                if batch_size:
                    X = X[:batch_size]
                    y = y[:batch_size]

                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                _, predicted = outputs.max(1)
                total_correct += (predicted == y).sum().item()
                total_samples += y.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, {"accuracy": accuracy}

    # -------------------------------------------------------------------------
    # Client count helpers
    # -------------------------------------------------------------------------

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.min_fit_clients, self.min_available_clients

    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.min_evaluate_clients, self.min_available_clients
