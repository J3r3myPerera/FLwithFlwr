"""
Federated Learning for Disposable Income Regression - Google Colab Standalone

This file contains all necessary code to run FL simulations on Google Colab.
Upload this file and the dataset CSV to Colab, then run the cells.

Features:
- Multiple FL strategies: FedAvg, FedProx, SCAFFOLD, Hybrid
- Log-scale transformation for better MAPE
- FL-optimized model with LayerNorm, GELU, Dropout
- AdamW optimizer with cosine annealing
- Non-IID data partitioning by City_Tier

Usage in Colab:
1. Upload this file and indianPersonalFinanceAndSpendingHabits.csv
2. Install dependencies: !pip install torch flwr pandas scikit-learn
3. Run: python colab_standalone.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Union
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Flower imports
import flwr as fl
from flwr.common import (
    Parameters, FitRes, EvaluateRes, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
    FitIns, EvaluateIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, FedAvg as FlowerFedAvg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters"""
    # Dataset
    DATA_PATH = './indianPersonalFinanceAndSpendingHabits.csv'
    
    # FL Training
    STRATEGY = "fedavg"  # Options: fedavg, fedprox, scaffold, hybrid
    NUM_ROUNDS = 30
    NUM_CLIENTS = 3
    LOCAL_EPOCHS = 8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Strategy-specific
    FEDPROX_MU = 0.05
    SCAFFOLD_LR_CORRECTION = 1.0
    HYBRID_FEDPROX_WEIGHT = 0.3
    HYBRID_SCAFFOLD_WEIGHT = 0.5
    HYBRID_MU = 0.03
    
    # Model
    HIDDEN_LAYERS = [128, 64, 32]
    DROPOUT = 0.15
    
    # Data
    IID = False  # Non-IID by City_Tier
    NORMALIZE_TARGET = True
    LOG_TRANSFORM_TARGET = True
    SEED = 2023


# =============================================================================
# MODEL
# =============================================================================

class DisposableIncomeNet(nn.Module):
    """FL-Optimized Neural Network with LayerNorm, GELU, Dropout"""
    
    def __init__(
        self, 
        input_dim: int = 25,
        hidden_dim1: int = 128, 
        hidden_dim2: int = 64,
        hidden_dim3: int = 32,
        dropout: float = 0.15,
        use_layer_norm: bool = True
    ):
        super(DisposableIncomeNet, self).__init__()
        self.use_layer_norm = use_layer_norm
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        if use_layer_norm:
            self.ln2 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        if use_layer_norm:
            self.ln3 = nn.LayerNorm(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dim3, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for GELU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if self.use_layer_norm:
            x = self.ln3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        return x


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as list of NumPy arrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List) -> None:
    """Set model parameters from list of NumPy arrays"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# =============================================================================
# DATASET
# =============================================================================

class DisposableIncomeDataset(Dataset):
    """PyTorch Dataset for Disposable Income Regression"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def compute_disposable_income(df: pd.DataFrame) -> pd.Series:
    """Compute target: Disposable_Income = Income - Total_Expenses"""
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                    'Education', 'Miscellaneous']
    total_expenses = df[expense_cols].sum(axis=1)
    return df['Income'] - total_expenses


class DataPreprocessor:
    """Data preprocessing with feature engineering"""
    
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.encoder = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.is_fitted = False
        self.log_transform_target = False
        self.target_shift = 0.0
        
        self.base_numerical_features = [
            'Age', 'Dependents',
            'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
            'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education'
        ]
        
        self.engineered_features = [
            'Total_Expenses', 'Expense_to_Income_Ratio',
            'Essential_Expenses', 'Discretionary_Expenses',
            'Age_squared', 'Income_log'
        ]
        
        self.numerical_features = self.base_numerical_features
        self.categorical_features = ['Occupation', 'City_Tier']
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        df = df.copy()
        
        expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                       'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education']
        df['Total_Expenses'] = df[expense_cols].sum(axis=1)
        df['Expense_to_Income_Ratio'] = df['Total_Expenses'] / (df['Income'] + 1)
        df['Essential_Expenses'] = df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']
        df['Discretionary_Expenses'] = df['Eating_Out'] + df['Entertainment']
        df['Age_squared'] = df['Age'] ** 2
        df['Income_log'] = np.log1p(df['Income'])
        
        return df
    
    def fit(self, df: pd.DataFrame, normalize_target: bool = True, log_transform_target: bool = False):
        """Fit preprocessor on training data"""
        df = self._engineer_features(df)
        self.numerical_features = self.base_numerical_features + self.engineered_features
        
        # Fit scaler
        X_numerical = df[self.numerical_features].values.astype(np.float32)
        self.numerical_scaler.fit(X_numerical)
        
        # Fit encoder
        try:
            self.encoder = OneHotEncoder(sparse_output=False, drop=None)
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, drop=None)
        self.encoder.fit(df[self.categorical_features])
        
        # Target with log transformation
        y = compute_disposable_income(df).values.astype(np.float32)
        
        self.log_transform_target = log_transform_target
        if log_transform_target:
            y_min = y.min()
            if y_min < 0:
                self.target_shift = -y_min + 1
                y = y + self.target_shift
            else:
                self.target_shift = 0.0
            y = np.log1p(y)
        
        self.target_mean = y.mean()
        self.target_std = y.std()
        self.normalize_target = normalize_target
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data"""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        
        df = self._engineer_features(df)
        
        # Transform features
        X_numerical = df[self.numerical_features].values.astype(np.float32)
        X_numerical = self.numerical_scaler.transform(X_numerical)
        X_categorical = self.encoder.transform(df[self.categorical_features])
        X = np.hstack([X_numerical, X_categorical]).astype(np.float32)
        
        # Transform target
        y = compute_disposable_income(df).values.astype(np.float32)
        
        if self.log_transform_target:
            if self.target_shift > 0:
                y = y + self.target_shift
            y = np.log1p(y)
        
        if self.normalize_target:
            y = (y - self.target_mean) / self.target_std
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame, normalize_target: bool = True, 
                      log_transform_target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform"""
        self.fit(df, normalize_target, log_transform_target)
        return self.transform(df)


def prepare_federated_data(
    data_path: str,
    num_clients: int = 3,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023,
    iid: bool = False,
    log_transform: bool = True
):
    """Prepare federated dataset"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load and preprocess
    df = pd.read_csv(data_path)
    print(f"[Data] Loaded {len(df)} samples")
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df, normalize_target=True, log_transform_target=log_transform)
    
    input_dim = X.shape[1]
    print(f"[Features] {input_dim} features")
    
    # Non-IID partition by City_Tier
    if not iid:
        tier_names = ['Tier_1', 'Tier_2', 'Tier_3']
        trainloaders, valloaders = [], []
        test_X_list, test_y_list = [], []
        
        print("[Partitioning] Non-IID by City_Tier")
        
        for tier in tier_names:
            mask = df['City_Tier'] == tier
            X_tier = X[mask]
            y_tier = y[mask]
            
            # Split
            tier_size = len(y_tier)
            tier_test_size = int(test_ratio * tier_size)
            tier_trainval_size = tier_size - tier_test_size
            
            indices = np.random.permutation(tier_size)
            trainval_indices = indices[:tier_trainval_size]
            test_indices = indices[tier_trainval_size:]
            
            # Create datasets
            client_dataset = DisposableIncomeDataset(X_tier[trainval_indices], y_tier[trainval_indices])
            
            # Split train/val
            val_size = max(1, int(val_ratio * len(client_dataset)))
            train_size = len(client_dataset) - val_size
            train_subset, val_subset = random_split(
                client_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
            
            # Collect test data
            test_X_list.append(X_tier[test_indices])
            test_y_list.append(y_tier[test_indices])
            
            print(f"  {tier}: {train_size} train, {val_size} val")
        
        # Create test set
        test_X = np.vstack(test_X_list)
        test_y = np.concatenate(test_y_list)
        test_dataset = DisposableIncomeDataset(test_X, test_y)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # IID partition (simple random split)
        raise NotImplementedError("IID partitioning not implemented in standalone version")
    
    return (trainloaders, valloaders, testloader, input_dim, 
            preprocessor.target_mean, preprocessor.target_std, log_transform)


# =============================================================================
# CLIENT
# =============================================================================

class RegressionClient(fl.client.NumPyClient):
    """Federated client with multiple strategy support"""
    
    def __init__(
        self,
        cid: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        input_dim: int,
        target_mean: float,
        target_std: float,
        log_transform: bool,
        local_epochs: int,
        learning_rate: float,
        strategy: str,
        mu: float,
        scaffold_lr_correction: float,
        fedprox_weight: float,
        scaffold_weight: float,
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
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Model
        self.model = DisposableIncomeNet(input_dim=input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        
        # For FedProx and Hybrid
        self.global_params = None
        
        # For SCAFFOLD and Hybrid
        self.client_control = None
        self.server_control = None
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters"""
        return get_parameters(self.model)
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters"""
        set_parameters(self.model, parameters)
        # Store global params for FedProx
        if self.strategy in ["fedprox", "hybrid"]:
            self.global_params = [torch.tensor(p).to(self.device) for p in parameters]
    
    def _init_control_variates(self):
        """Initialize SCAFFOLD control variates"""
        self.client_control = [torch.zeros_like(p) for p in self.model.parameters()]
        self.server_control = [torch.zeros_like(p) for p in self.model.parameters()]
    
    def _train_fedavg(self) -> float:
        """FedAvg training with AdamW"""
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_fedprox(self) -> float:
        """FedProx training with proximal term"""
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
                
                # Proximal term
                proximal_term = 0.0
                if self.global_params is not None:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p) ** 2)
                    proximal_term = (self.mu / 2.0) * proximal_term
                
                loss = mse_loss + proximal_term
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _train_scaffold(self) -> Tuple[float, List[np.ndarray]]:
        """SCAFFOLD training with control variates"""
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
                
                # SCAFFOLD correction
                with torch.no_grad():
                    for param, c_i, c in zip(self.model.parameters(), 
                                              self.client_control, 
                                              self.server_control):
                        if param.grad is not None:
                            param.grad.add_(c.to(self.device) - c_i.to(self.device))
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
                total_steps += 1
        
        # Update client control
        old_client_control = [c.clone() for c in self.client_control]
        
        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control, 
                initial_params, self.model.parameters()
            )):
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / (total_steps * self.learning_rate)
                self.client_control[i] = c_i.to(self.device) - c.to(self.device) + param_diff
        
        delta_control = [
            (new_c - old_c).cpu().numpy() 
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]
        
        return total_loss / n_batches if n_batches > 0 else 0.0, delta_control
    
    def _train_hybrid(self) -> Tuple[float, List[np.ndarray]]:
        """Hybrid FedProx + SCAFFOLD training"""
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
                mse_loss = self.criterion(y_pred, y_batch)
                
                # FedProx proximal term (weighted)
                proximal_term = 0.0
                if self.global_params is not None and self.fedprox_weight > 0:
                    for local_p, global_p in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_p - global_p.to(self.device)) ** 2)
                    proximal_term = (self.fedprox_weight * self.mu / 2.0) * proximal_term
                
                loss = mse_loss + proximal_term
                
                optimizer.zero_grad()
                loss.backward()
                
                # SCAFFOLD correction (weighted)
                if self.scaffold_weight > 0:
                    with torch.no_grad():
                        for param, c_i, c in zip(self.model.parameters(),
                                                  self.client_control,
                                                  self.server_control):
                            if param.grad is not None:
                                correction = self.scaffold_weight * (c.to(self.device) - c_i.to(self.device))
                                param.grad.add_(correction)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                n_batches += 1
                total_steps += 1
        
        # Update client control
        old_client_control = [c.clone() for c in self.client_control]
        
        with torch.no_grad():
            for i, (c_i, c, x_0, x_K) in enumerate(zip(
                self.client_control, self.server_control,
                initial_params, self.model.parameters()
            )):
                param_diff = (x_0.to(self.device) - x_K.to(self.device)) / (total_steps * self.learning_rate)
                self.client_control[i] = c_i.to(self.device) - c.to(self.device) + param_diff
        
        delta_control = [
            (new_c - old_c).cpu().numpy()
            for new_c, old_c in zip(self.client_control, old_client_control)
        ]
        
        return total_loss / n_batches if n_batches > 0 else 0.0, delta_control
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model"""
        self.set_parameters(parameters)
        
        # Update config
        self.local_epochs = int(config.get("local_epochs", self.local_epochs))
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.mu = float(config.get("mu", self.mu))
        
        # Train based on strategy
        delta_control = None
        
        if self.strategy == "fedavg":
            loss = self._train_fedavg()
        elif self.strategy == "fedprox":
            loss = self._train_fedprox()
        elif self.strategy == "scaffold":
            loss, delta_control = self._train_scaffold()
        elif self.strategy == "hybrid":
            loss, delta_control = self._train_hybrid()
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
        
        return new_parameters, num_samples, metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model"""
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
        
        # Denormalize
        preds_original = all_preds * self.target_std + self.target_mean
        targets_original = all_targets * self.target_std + self.target_mean
        
        # Inverse log transform
        if self.log_transform:
            preds_original = torch.expm1(preds_original)
            targets_original = torch.expm1(targets_original)
        
        # Metrics
        avg_loss = total_loss / num_samples
        mae = torch.mean(torch.abs(preds_original - targets_original)).item()
        
        return float(avg_loss), num_samples, {"mae": mae}


def create_client(cid: str, trainloader, valloader, input_dim, target_mean, target_std, 
                  log_transform, config) -> RegressionClient:
    """Factory function to create client"""
    return RegressionClient(
        cid=cid,
        trainloader=trainloader,
        valloader=valloader,
        input_dim=input_dim,
        target_mean=target_mean,
        target_std=target_std,
        log_transform=log_transform,
        local_epochs=config.LOCAL_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        strategy=config.STRATEGY,
        mu=config.FEDPROX_MU,
        scaffold_lr_correction=config.SCAFFOLD_LR_CORRECTION,
        fedprox_weight=config.HYBRID_FEDPROX_WEIGHT,
        scaffold_weight=config.HYBRID_SCAFFOLD_WEIGHT
    )


# =============================================================================
# SERVER
# =============================================================================

def compute_metrics(model, dataloader, target_std, target_mean, device, log_transform):
    """Compute regression metrics"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    num_samples = len(all_targets)
    
    # Denormalize
    preds_original = all_preds * target_std + target_mean
    targets_original = all_targets * target_std + target_mean
    
    # Inverse log transform
    if log_transform:
        preds_original = torch.expm1(preds_original)
        targets_original = torch.expm1(targets_original)
    
    # Metrics
    avg_loss = total_loss / num_samples
    mse = torch.mean((preds_original - targets_original) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(preds_original - targets_original)).item()
    
    ss_res = torch.sum((targets_original - preds_original) ** 2).item()
    ss_tot = torch.sum((targets_original - torch.mean(targets_original)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    epsilon = 1e-8
    mape = torch.mean(torch.abs((targets_original - preds_original) / (targets_original + epsilon))) * 100
    mape = mape.item()
    
    # Accuracy thresholds
    threshold_10 = torch.abs((targets_original - preds_original) / (targets_original + epsilon)) < 0.10
    threshold_20 = torch.abs((targets_original - preds_original) / (targets_original + epsilon)) < 0.20
    accuracy_10 = threshold_10.float().mean().item() * 100
    accuracy_20 = threshold_20.float().mean().item() * 100
    
    return {
        "loss": avg_loss,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "accuracy_10": accuracy_10,
        "accuracy_20": accuracy_20
    }


class FedAvgStrategy(FlowerFedAvg):
    """FedAvg strategy with custom evaluation"""
    
    def __init__(self, testloader, input_dim, target_mean, target_std, log_transform, model, device, **kwargs):
        super().__init__(**kwargs)
        self.testloader = testloader
        self.input_dim = input_dim
        self.target_mean = target_mean
        self.target_std = target_std
        self.log_transform = log_transform
        self.model = model
        self.device = device
    
    def evaluate(self, server_round, parameters):
        """Evaluate global model"""
        set_parameters(self.model, parameters_to_ndarrays(parameters))
        metrics = compute_metrics(
            self.model, self.testloader, self.target_std, 
            self.target_mean, self.device, self.log_transform
        )
        
        print(f"Round {server_round:2d} | Loss: {metrics['loss']:.4f} | "
              f"MAPE: {metrics['mape']:5.2f}% | R²: {metrics['r2']:.4f} | "
              f"MAE: ${metrics['mae']:,.0f}")
        
        return metrics["loss"], metrics


# =============================================================================
# MAIN
# =============================================================================

def run_simulation(config: Config):
    """Run FL simulation"""
    print("\n" + "=" * 70)
    print(f"Federated Learning - {config.STRATEGY.upper()}")
    print("=" * 70)
    
    # Prepare data
    print("\n[1/4] Preparing federated dataset...")
    (trainloaders, valloaders, testloader, input_dim, 
     target_mean, target_std, log_transform) = prepare_federated_data(
        data_path=config.DATA_PATH,
        num_clients=config.NUM_CLIENTS,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        iid=config.IID,
        log_transform=config.LOG_TRANSFORM_TARGET
    )
    
    # Create model
    print("\n[2/4] Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisposableIncomeNet(input_dim=input_dim).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    
    # Initialize parameters
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    
    # Create strategy
    print("\n[3/4] Creating strategy...")
    
    def on_fit_config(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": config.LOCAL_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "mu": config.FEDPROX_MU if config.STRATEGY == "fedprox" else config.HYBRID_MU
        }
    
    strategy = FedAvgStrategy(
        testloader=testloader,
        input_dim=input_dim,
        target_mean=target_mean,
        target_std=target_std,
        log_transform=log_transform,
        model=model,
        device=device,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_config,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=config.NUM_CLIENTS,
        min_available_clients=config.NUM_CLIENTS
    )
    
    # Create client function
    def client_fn(cid: str) -> fl.client.Client:
        return create_client(
            cid=cid,
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            input_dim=input_dim,
            target_mean=target_mean,
            target_std=target_std,
            log_transform=log_transform,
            config=config
        )
    
    # Run simulation
    print("\n[4/4] Starting simulation...")
    print(f"Strategy: {config.STRATEGY}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Local epochs: {config.LOCAL_EPOCHS}")
    print("-" * 70)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Final metrics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    final_metrics = history.metrics_distributed
    if final_metrics:
        last_round = max(final_metrics.keys())
        for metric_name, values in final_metrics.items():
            if values:
                print(f"{metric_name}: {values[-1][1]:.4f}")
    
    return history


if __name__ == "__main__":
    # Create config
    config = Config()
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_PATH):
        print(f"\n❌ ERROR: Dataset not found at {config.DATA_PATH}")
        print("Please upload indianPersonalFinanceAndSpendingHabits.csv to the current directory")
        sys.exit(1)
    
    # Run simulation
    history = run_simulation(config)
    
    print("\n✅ Simulation completed successfully!")
    print("\nTo change strategy, modify Config.STRATEGY:")
    print("  - 'fedavg': Standard federated averaging")
    print("  - 'fedprox': With proximal term (better for non-IID)")
    print("  - 'scaffold': With control variates (best for non-IID)")
    print("  - 'hybrid': Combined FedProx + SCAFFOLD (maximum robustness)")
