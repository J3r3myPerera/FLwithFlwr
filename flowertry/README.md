# Flower Federated Learning Project

## Project Overview
This project is a complete Federated Learning implementation using the Flower framework with MNIST dataset. It includes client-server architecture, neural network model, and data partitioning for federated learning experiments.

**Key Feature: IID vs Non-IID Data Simulation** - This project supports comparing how federated learning performs under ideal (IID) vs realistic (non-IID) data distributions, with FedProx as a mitigation strategy. Includes both moderate and **extreme non-IID** configurations to clearly demonstrate the impact of data heterogeneity.

## Quick Start: Running Experiments

### Standard Experiments (Moderate Non-IID)
```bash
# Run IID baseline (ideal scenario)
python main.py --config-name=iid_fedavg

# Run non-IID without mitigation (demonstrates the problem)
python main.py --config-name=noniid_fedavg

# Run non-IID with FedProx (demonstrates the solution)
python main.py --config-name=noniid_fedprox
```

### Extreme Non-IID Experiments (Recommended for Clear Demonstration)
```bash
# Extreme non-IID without mitigation (shows severe accuracy drop)
python main.py --config-name=extreme_noniid_fedavg

# Extreme non-IID with tuned FedProx (shows significant recovery)
python main.py --config-name=extreme_noniid_fedprox

# Compare all results
python visualize.py
```

## Non-IID Data Simulation

### What is Non-IID Data?
In real-world federated learning, data on different clients is often **non-IID** (non-Independent and Identically Distributed):
- A hospital in one region may see different diseases than another
- User behavior data varies significantly across demographics
- Mobile keyboard data reflects different languages/typing styles

### Label Skew with Dirichlet Distribution
This project uses the **Dirichlet distribution** to simulate label skew:
- **α = 0.1**: Extreme non-IID (each client gets mostly 1-2 classes) - **recommended for clear demonstration**
- **α = 0.5**: Moderate non-IID
- **α → ∞**: Approaches IID (uniform distribution)

### FedProx Mitigation
FedProx adds a proximal term to prevent client drift:
```
L_local = L_original + (μ/2) * ||w - w_global||²
```
This penalizes local models that diverge too far from the global model.

### Experiment Configurations

| Config | Partition | Strategy | Local Epochs | Description |
|--------|-----------|----------|--------------|-------------|
| `iid_fedavg.yaml` | IID | FedAvg | 3 | Baseline - best case scenario |
| `noniid_fedavg.yaml` | Dirichlet (α=0.5) | FedAvg | 1 | Moderate non-IID |
| `noniid_fedprox.yaml` | Dirichlet (α=0.5) | FedProx (μ=0.1) | 1 | Moderate non-IID with mitigation |
| `extreme_noniid_fedavg.yaml` | Dirichlet (α=0.1) | FedAvg | 3 | **Extreme non-IID - shows the problem** |
| `extreme_noniid_fedprox.yaml` | Dirichlet (α=0.1) | FedProx (μ=0.5) | 3 | **Extreme non-IID - shows the solution** |

### Expected Results

#### Standard Experiments (α=0.5, 1 local epoch)
- **IID + FedAvg**: ~97-98% accuracy, fast convergence
- **Non-IID + FedAvg**: ~90-94% accuracy, slower convergence
- **Non-IID + FedProx**: ~93-96% accuracy, recovered stability

#### Extreme Experiments (α=0.1, 3 local epochs) - **Recommended**
- **IID + FedAvg**: ~95-97% accuracy, stable convergence
- **Extreme Non-IID + FedAvg**: ~70-85% accuracy, unstable, may oscillate
- **Extreme Non-IID + FedProx**: ~88-93% accuracy, significant recovery

## Visualization Tools

```bash
# Compare experiment results (after running experiments)
python visualize.py

# Visualize IID vs non-IID data distribution
python visualize.py --show_distribution --alpha 0.1

# See how different alpha values affect heterogeneity
python visualize.py --show_alpha_sensitivity
```

## Recent Changes (Latest Updates)

### Non-IID Simulation Framework (December 2025)

#### New Features
- **Dirichlet-based Non-IID Partitioning**: Implemented `partition_dirichlet()` function in `dataset.py` to simulate realistic label skew across clients
- **FedProx Implementation**: Added `FedProxClient` class in `cleint.py` with proximal term regularization to mitigate client drift
- **Extreme Non-IID Experiments**: New configurations (`extreme_noniid_fedavg.yaml`, `extreme_noniid_fedprox.yaml`) with α=0.1 and 3 local epochs for dramatic demonstration of non-IID effects
- **Visualization Script**: Added `visualize.py` for comparing experiment results, plotting accuracy curves, and visualizing data distributions
- **Dynamic Hardware Detection**: `main.py` now auto-detects CPU cores and optimizes client resource allocation

#### Hardware Optimization for M1 Pro MacBook
- **CPU-only Training**: Disabled MPS (Apple Silicon GPU) due to incompatibility with Ray's multiprocessing used by Flower simulation
- **Smart Resource Allocation**: Dynamically calculates CPUs per client based on available cores and clients per round
- **Device Detection**: Unified `get_device()` function in `cleint.py` and `server.py` for consistent device selection

#### Configuration Files Added
- `conf/iid_fedavg.yaml` - IID baseline with 3 local epochs
- `conf/noniid_fedavg.yaml` - Moderate non-IID (α=0.5)
- `conf/noniid_fedprox.yaml` - Moderate non-IID with FedProx (μ=0.1)
- `conf/extreme_noniid_fedavg.yaml` - Extreme non-IID (α=0.1, 3 epochs)
- `conf/extreme_noniid_fedprox.yaml` - Extreme non-IID with tuned FedProx (μ=0.5)

#### Bug Fixes
- **Fixed `model.py`**: Added `net.to(device)` in `test()` function to prevent device mismatch errors
- **Fixed `server.py`**: Added `get_device()` function for proper device handling
- **Fixed `cleint.py`**: Added `get_device()` function with CUDA > CPU fallback (MPS disabled for Ray compatibility)

### Previous Fixes and Enhancements
- **Fixed `server.py`**: Added missing `test` import from `model` module
- **Fixed `model.py`**: Corrected dimension mismatch in `fc2` layer
- **Enhanced `main.py`**: Added result saving functionality and comprehensive experiment logging
- **Fixed `cleint.py`**: Resolved type imports and method call issues

### Complete Federated Learning Implementation
- **Added `model.py`**: Neural network architecture (CNN) for MNIST classification
- **Added `cleint.py`**: Flower client implementation with training and evaluation
- **Enhanced `main.py`**: Integrated all components with proper workflow
- **Updated `conf/base.yaml`**: Added comprehensive configuration parameters
- **Fixed `dataset.py`**: Resolved partitioning issues and improved data handling

### New Files Added

#### 1. `model.py` - Neural Network Architecture
- **CNN Model**: Convolutional Neural Network for MNIST digit classification
- **Architecture**: 
  - Conv2d layers: 1→6→16 channels
  - MaxPool2d: 2x2 pooling
  - Fully connected layers: 256→120→84→num_classes (fixed dimension mismatch)
- **Training Function**: `train()` with CrossEntropyLoss and SGD optimizer
- **Testing Function**: `test()` with accuracy calculation
- **Device Support**: Automatic GPU/CPU detection
- **Bug Fix**: Fixed `fc2` layer input dimension from 128 to 120 to match `fc1` output

#### 2. `cleint.py` - Flower Client Implementation
- **FlowerClient Class**: Implements `fl.client.NumPyClient`
- **Type Imports**: Properly imports `Dict` from `typing` and `Scalar`, `NDArrays` from `flwr.common.typing`
- **Key Methods**:
  - `set_parameters()`: Load server parameters into local model
  - `get_parameters()`: Extract model parameters for server (accepts optional config)
  - `fit()`: Local training with configurable hyperparameters (correctly calls `set_parameters`)
  - `evaluate()`: Local validation with loss and accuracy metrics (correctly calls `set_parameters`)
- **Client Factory**: `generate_client_fn()` for creating multiple clients
- **Configuration Support**: Dynamic learning rate, momentum, and epochs
- **Bug Fixes**: Fixed assignment operator and method call issues

#### 3. Enhanced `main.py`
- **Complete Workflow**: Dataset preparation → Client generation → Server strategy setup → Simulation execution → Result saving
- **Import Integration**: Proper imports for all modules including Flower, server functions, Hydra, and pickle
- **Server Strategy**: FedAvg strategy configuration with fit/evaluate parameters
- **Simulation Execution**: Full federated learning simulation with `fl.simulation.start_simulation()`
- **Resource Management**: Optimized `client_resources` for M1 Pro MacBook (10 CPU cores) - allocates 0.8 CPUs per client, 0 GPUs (MPS handled separately)
- **Result Persistence**: Automatically saves training history to `results.pkl` in Hydra output directory
- **Debugging Output**: Client count and dataset size information
- **Configuration Integration**: Uses Hydra config for all parameters

#### 4. Added `server.py` - Server Configuration Functions
- **`get_on_fit_config()`**: Configures hyperparameters (lr, momentum, local_epochs) for each federated round
- **`get_evaluate_fn()`**: Global model evaluation function for server-side testing
- **Model Loading**: Loads aggregated parameters into model for evaluation
- **Test Set Evaluation**: Returns loss and accuracy metrics on global test set
- **Import Fix**: Added `test` function import from `model` module to enable server-side evaluation

#### 5. Updated `conf/base.yaml`
- **New Parameters**:
  - `batch_size: 20` - Training batch size
  - `num_classes: 10` - MNIST digit classes
  - `config_fit` - Training configuration:
    - `lr: 0.01` - Learning rate
    - `momentum: 0.9` - SGD momentum
    - `local_epochs: 1` - Local training epochs per round

#### 6. Fixed `dataset.py`
- **Partitioning Fix**: Corrected data splitting logic
- **Simplified Approach**: Direct division without remainder handling
- **IID Partitioning**: Equal data distribution among clients
- **Validation Split**: 10% validation data per client

## Project Structure
```
flowertry/
├── conf/
│   ├── base.yaml                    # Base configuration file
│   ├── iid_fedavg.yaml              # IID + FedAvg baseline
│   ├── noniid_fedavg.yaml           # Moderate non-IID + FedAvg
│   ├── noniid_fedprox.yaml          # Moderate non-IID + FedProx
│   ├── extreme_noniid_fedavg.yaml   # Extreme non-IID + FedAvg (α=0.1)
│   └── extreme_noniid_fedprox.yaml  # Extreme non-IID + FedProx (α=0.1, μ=0.5)
├── main.py                # Main application entry point
├── dataset.py             # Dataset handling with IID/Dirichlet non-IID partitioning
├── model.py               # Neural network architecture (CNN)
├── cleint.py              # Flower client (FlowerClient + FedProxClient)
├── server.py              # Server configuration functions
├── visualize.py           # Results visualization and comparison
├── outputs/               # Experiment outputs (Hydra-managed)
│   └── YYYY-MM-DD/       # Date-based organization
│       └── HH-MM-SS/     # Time-based experiment folder
│           ├── .hydra/    # Hydra configuration files
│           ├── main.log   # Application logs
│           └── results.pkl # Saved training history
├── plots/                 # Generated visualization plots
├── data/                  # MNIST dataset (auto-downloaded)
└── README.md              # This file
```

## Current Implementation Status

### ✅ Completed Features
- **Data Pipeline**: MNIST loading, preprocessing, IID and Dirichlet non-IID partitioning
- **Neural Network**: CNN model for digit classification
- **Client Implementation**: FlowerClient (FedAvg) and FedProxClient with proximal regularization
- **Server Functions**: Server-side configuration and evaluation functions
- **Federated Strategies**: FedAvg and FedProx implemented
- **Non-IID Simulation**: Dirichlet distribution-based label skew simulation
- **Extreme Experiments**: Configurations for demonstrating severe non-IID effects
- **Visualization**: Accuracy comparison plots, data distribution heatmaps
- **Hardware Optimization**: Dynamic CPU allocation for M1 Pro MacBook
- **Result Persistence**: Training history automatically saved to pickle file
- **Configuration Management**: Hydra-based parameter configuration with multiple experiment profiles

### 🔧 Known Issues
- **Filename typo**: `cleint.py` should be `client.py` (cosmetic, functionality works)
- **MPS Disabled**: Apple Silicon GPU not used due to Ray multiprocessing incompatibility

### 🚀 Potential Enhancements
- **SCAFFOLD Implementation**: More sophisticated variance reduction algorithm
- **Adaptive FedProx**: Dynamically adjust μ based on client divergence
- **More Datasets**: Extend to CIFAR-10, FEMNIST, or custom datasets
- **Cross-device Simulation**: Simulate device heterogeneity (different compute capabilities)

## Environment Setup
- **Python Environment**: `flower_tutorial` conda environment
- **Frameworks**: 
  - Flower (Federated Learning)
  - PyTorch (Deep Learning)
  - Torchvision (Computer Vision)
- **Configuration**: Hydra
- **IDE**: VS Code with Python extension

## Running the Project
To run the current setup:
```bash
python main.py
```

This will:
1. Load and display configuration
2. Prepare MNIST dataset with IID partitioning
3. Generate client functions for federated learning
4. Display dataset statistics
5. Start federated learning simulation with configured strategy
6. Execute training rounds with client selection
7. Evaluate global model on test set after each round
8. Save training history to `results.pkl` in output directory

## Configuration Parameters

### Base Parameters
- `num_rounds: 10` - Federated learning rounds
- `num_clients: 100` - Number of participating clients
- `batch_size: 20` - Training batch size
- `num_classes: 10` - MNIST digit classes (0-9)
- `num_clients_per_round_fit: 10` - Minimum clients selected for training per round
- `num_clients_per_round_eval: 25` - Minimum clients selected for evaluation per round

### Data Partitioning Parameters
- `partition_type: "iid"` - Partitioning strategy ("iid" or "dirichlet")
- `dirichlet_alpha` - Dirichlet concentration parameter:
  - `0.1` - Extreme non-IID (each client gets mostly 1-2 classes) **recommended for demonstration**
  - `0.5` - Moderate non-IID
  - `1.0+` - Mild non-IID, approaching IID

### Strategy Parameters
- `strategy: "fedavg"` - Aggregation strategy ("fedavg" or "fedprox")
- `fedprox_mu` - FedProx proximal coefficient:
  - `0.1` - Light regularization (for moderate non-IID)
  - `0.5` - Strong regularization (for extreme non-IID) **recommended**
  - `1.0+` - Very strong regularization

### Training Parameters
- `config_fit` - Local training parameters:
  - `lr: 0.01` - Learning rate
  - `momentum: 0.9` - SGD momentum
  - `local_epochs` - Local training epochs (higher = more client drift)
    - `1` - Standard setting
    - `3` - Recommended for extreme non-IID experiments (amplifies drift)

## Next Steps
1. **Run Extreme Experiments**: Execute `extreme_noniid_fedavg` and `extreme_noniid_fedprox` to see dramatic differences
2. **Tune FedProx**: Experiment with different μ values (0.1, 0.5, 1.0) to find optimal settings
3. **Vary Alpha**: Test different Dirichlet α values to understand heterogeneity impact
4. **Implement SCAFFOLD**: Add more advanced variance reduction for even better non-IID handling
5. **Extend to Other Datasets**: Apply framework to CIFAR-10 or real-world federated datasets

## Technical Notes

### Data Partitioning
- **IID**: Random split with equal samples per client
- **Dirichlet Non-IID**: Label skew controlled by α parameter - lower values create more extreme heterogeneity

### Model Architecture
- CNN suitable for 28x28 grayscale MNIST images
- Conv layers: 1→6→16 channels with 5x5 kernels
- Fully connected layers: 256→120→84→num_classes

### FedProx vs FedAvg
- **FedAvg**: Standard federated averaging - struggles with non-IID data
- **FedProx**: Adds proximal term `(μ/2) * ||w - w_global||²` to prevent client drift

### Hardware (M1 Pro MacBook)
- **CPU-only Training**: MPS (Apple Silicon GPU) is **disabled** due to incompatibility with Ray's multiprocessing
- **Dynamic Resource Allocation**: Auto-detects CPU cores and allocates optimally per client
- **Still Efficient**: M1 Pro's unified memory and high single-core performance provide good CPU training speed

### Result Storage
- Training history saved as pickle file (`results.pkl`) in Hydra-managed output directory
- Contains accuracy/loss curves and experiment configuration for later analysis
