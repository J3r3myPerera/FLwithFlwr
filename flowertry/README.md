# Flower Federated Learning Project

## Project Overview
This project is a complete Federated Learning implementation using the Flower framework with MNIST dataset. It supports **non-IID data partitioning** and multiple aggregation strategies (**FedAvg** and **FedProx**) for realistic federated learning experiments.

## Recent Changes (Latest Updates - December 2025)

### 🆕 Non-IID Data Partitioning Support
- **Dirichlet Distribution**: Realistic non-IID partitioning with configurable heterogeneity (alpha parameter)
- **Label Skew**: Each client has limited number of classes
- **Quantity Skew**: Unbalanced data sizes per client
- **IID Baseline**: Uniform random split for comparison
- **Partition Statistics**: Automatic printing of data distribution per client

### 🆕 FedProx Strategy Implementation
- **Proximal Regularization**: Prevents client drift in non-IID settings
- **Configurable mu**: Tunable proximal term coefficient
- **Strategy Selection**: Easy switching between FedAvg and FedProx via config

### 🆕 Apple Silicon (M1/M2) Optimization
- **Reduced Scale**: 10 clients (down from 100) for faster local simulation
- **CPU-only Mode**: MPS disabled due to Ray/Flower compatibility issues
- **macOS Compatibility**: `num_workers=0` in DataLoaders to avoid multiprocessing issues
- **Optimized Resources**: 2 CPUs per client for M1 Pro (10 cores)

### 🆕 Configuration Presets
- `base.yaml`: Default configuration with FedProx and non-IID
- `fedavg_baseline.yaml`: FedAvg baseline for comparison
- `fedprox_tuned.yaml`: Optimized FedProx configuration

### Bug Fixes
- **Fixed `model.py`**: Added `net.to(device)` in `test()` function to prevent device mismatch
- **Renamed `cleint.py` → `client.py`**: Fixed filename typo
- **Device Handling**: Proper device detection for CUDA/CPU (MPS disabled for Ray compatibility)

## Project Structure
```
flowertry/
├── conf/
│   ├── base.yaml              # Default configuration (FedProx + non-IID)
│   ├── fedavg_baseline.yaml   # FedAvg comparison config
│   └── fedprox_tuned.yaml     # Optimized FedProx config
├── main.py                    # Main application entry point
├── dataset.py                 # Dataset handling with non-IID partitioning
├── model.py                   # CNN architecture for MNIST
├── client.py                  # Flower client with FedProx support
├── server.py                  # Server strategy configuration
├── data/                      # MNIST dataset (auto-downloaded)
├── outputs/                   # Experiment outputs (Hydra-managed)
└── README.md                  # This file
```

## Non-IID Data Partitioning

### Dirichlet Distribution (Recommended)
Controls data heterogeneity with alpha parameter:
| Alpha | Heterogeneity | Description |
|-------|---------------|-------------|
| 0.1   | Extreme       | Each client dominated by 1-2 classes |
| 0.3   | High          | Significant class imbalance |
| 0.5   | Moderate      | Noticeable heterogeneity |
| 1.0   | Mild          | Slight imbalance |
| 10.0  | Near-IID      | Almost uniform distribution |

### Label Skew
Each client has exactly N classes:
- `labels_per_client=1`: Extreme (single class per client)
- `labels_per_client=2`: High non-IID
- `labels_per_client=5`: Moderate
- `labels_per_client=10`: IID-like

## Aggregation Strategies

### FedAvg (Baseline)
Standard federated averaging - simple parameter aggregation.

### FedProx (Non-IID Optimized)
Adds proximal regularization term to prevent client drift:
```
Loss = CrossEntropy + (μ/2) × ||w_local - w_global||²
```

**When FedProx outperforms FedAvg:**
- High data heterogeneity (α ≤ 0.1)
- Many local epochs (≥ 5)
- Properly tuned mu (0.001 recommended)

## Configuration Parameters

### Core Settings
```yaml
num_rounds: 20              # Federated learning rounds
num_clients: 10             # Total participating clients
num_clients_per_round_fit: 5    # Clients per training round
num_clients_per_round_eval: 5   # Clients per evaluation round
batch_size: 32              # Training batch size
num_classes: 10             # MNIST classes (0-9)
```

### Non-IID Settings
```yaml
partition_type: "dirichlet"  # iid, dirichlet, label_skew, quantity_skew
dirichlet_alpha: 0.1         # Lower = more non-IID
labels_per_client: 2         # For label_skew mode
```

### Strategy Settings
```yaml
strategy: "fedprox"          # fedavg or fedprox
fedprox_mu: 0.001            # Proximal term (0.001-0.01 recommended)
```

### Training Settings
```yaml
config_fit:
  lr: 0.01                   # Learning rate
  momentum: 0.9              # SGD momentum
  local_epochs: 5            # Local training epochs per round
```

## Running Experiments

### Default (FedProx + Non-IID)
```bash
python main.py
```

### Using Config Presets
```bash
# FedAvg baseline
python main.py --config-name=fedavg_baseline

# FedProx tuned
python main.py --config-name=fedprox_tuned
```

### Override Parameters
```bash
# Change non-IID level
python main.py dirichlet_alpha=0.5

# Compare strategies
python main.py strategy=fedavg
python main.py strategy=fedprox fedprox_mu=0.001

# Quick test
python main.py num_rounds=5

# IID baseline
python main.py partition_type=iid strategy=fedavg
```

### Hyperparameter Tuning
```bash
# Find optimal mu for FedProx
python main.py fedprox_mu=0.0001
python main.py fedprox_mu=0.001   # Recommended
python main.py fedprox_mu=0.01
python main.py fedprox_mu=0.1     # Usually too strong
```

## Model Architecture
CNN optimized for MNIST (28×28 grayscale):
```
Conv2d(1→6, 5×5) → ReLU → MaxPool(2×2)
Conv2d(6→16, 5×5) → ReLU → MaxPool(2×2)
Flatten → FC(256→120) → ReLU
FC(120→84) → ReLU → FC(84→10)
```

## Expected Results

### With Extreme Non-IID (α=0.1, 5 local epochs)
| Strategy | Round 10 Acc | Round 20 Acc | Notes |
|----------|--------------|--------------|-------|
| FedAvg   | ~94%         | ~96%         | May oscillate |
| FedProx  | ~95%         | ~97%         | More stable convergence |

### Partition Statistics Output
```
============================================================
PARTITION STATISTICS
============================================================
Number of clients: 10
Total samples: 60000
Samples per client - Min: 4521, Max: 7893, Mean: 6000.0

Label distribution (first 5 clients):
  Client 0: 6234 samples | 0:2341 3:1892 7:2001
  Client 1: 5123 samples | 1:3456 4:1667
  ...
============================================================
```

## Environment Setup

### Requirements
- Python 3.8+
- PyTorch
- Flower (flwr)
- Hydra
- NumPy
- torchvision

### Installation
```bash
conda create -n flower_tutorial python=3.10
conda activate flower_tutorial
pip install flwr torch torchvision hydra-core omegaconf
```

## Technical Notes

### Apple Silicon Compatibility
- **MPS Disabled**: Ray (used by Flower simulation) causes segfaults with MPS
- **CPU Mode**: Still performant for MNIST (~2-5 min for 20 rounds)
- **num_workers=0**: Avoids multiprocessing issues on macOS

### Device Selection Priority
1. CUDA (if available)
2. CPU (default for Flower simulation)
3. MPS (disabled due to Ray incompatibility)

### Result Storage
- Training history saved as `results.pkl` in Hydra output directory
- Contains: history, config, partition_type, strategy, alpha

## Troubleshooting

### Segmentation Fault on macOS
MPS + Ray incompatibility. Solution: Use CPU (already configured).

### "Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same"
Model not moved to device. Fixed in latest `model.py`.

### FedProx performing worse than FedAvg
- Lower mu (try 0.001 instead of 0.01)
- Increase local_epochs (≥5)
- Use more extreme non-IID (alpha ≤ 0.1)

## References
- [Flower Framework](https://flower.dev/)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)
- [Non-IID Data in FL](https://arxiv.org/abs/1806.00582)
