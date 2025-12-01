# Flower Federated Learning Project

## Project Overview
This project is a complete Federated Learning implementation using the Flower framework with MNIST dataset. It includes client-server architecture, neural network model, and data partitioning for federated learning experiments.

## Recent Changes (Latest Updates)

### Latest Fixes and Enhancements (Post-Last Git Push)
- **Fixed `cleint.py`**: Resolved `NameError` by adding missing type imports (`Dict`, `Scalar`, `NDArrays`)
- **Fixed `cleint.py`**: Corrected assignment operator (`==` → `=`) for model initialization
- **Fixed `cleint.py`**: Corrected method calls (`get_parameters` → `set_parameters`) in `fit()` and `evaluate()` methods
- **Fixed `cleint.py`**: Updated `get_parameters()` method signature to accept optional `config` parameter
- **Added `server.py`**: Server-side functions for federated learning strategy configuration
- **Enhanced `main.py`**: Added Flower server strategy (FedAvg) with configuration
- **Updated `conf/base.yaml`**: Added `num_clients_per_round_fit` and `num_clients_per_round_eval` parameters

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
  - Fully connected layers: 256→120→84→num_classes
- **Training Function**: `train()` with CrossEntropyLoss and SGD optimizer
- **Testing Function**: `test()` with accuracy calculation
- **Device Support**: Automatic GPU/CPU detection

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
- **Complete Workflow**: Dataset preparation → Client generation → Server strategy setup
- **Import Integration**: Proper imports for all modules including Flower and server functions
- **Server Strategy**: FedAvg strategy configuration with fit/evaluate parameters
- **Debugging Output**: Client count and dataset size information
- **Configuration Integration**: Uses Hydra config for all parameters

#### 4. Added `server.py` - Server Configuration Functions
- **`get_on_fit_config()`**: Configures hyperparameters (lr, momentum, local_epochs) for each federated round
- **`get_evaluate_fn()`**: Global model evaluation function for server-side testing
- **Model Loading**: Loads aggregated parameters into model for evaluation
- **Test Set Evaluation**: Returns loss and accuracy metrics on global test set

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
│   └── base.yaml          # Configuration file
├── main.py                # Main application entry point
├── dataset.py             # Dataset handling and partitioning
├── model.py               # Neural network architecture
├── cleint.py              # Flower client implementation
├── server.py              # Server configuration functions (NEW)
├── outputs/               # Experiment outputs
│   └── 2025-10-24/       # Date-based organization
│       └── 00-40-15/     # Time-based experiment folder
│           ├── .hydra/    # Hydra configuration files
│           └── main.log   # Application logs
├── .vscode/
│   └── settings.json      # VS Code workspace settings
└── README.md              # This file
```

## Current Implementation Status

### ✅ Completed Features
- **Data Pipeline**: MNIST loading, preprocessing, and IID partitioning
- **Neural Network**: CNN model for digit classification
- **Client Implementation**: Complete Flower client with training/evaluation (all bugs fixed)
- **Server Functions**: Server-side configuration and evaluation functions
- **Server Strategy**: FedAvg strategy configured in main.py
- **Configuration Management**: Hydra-based parameter configuration
- **Data Splitting**: Fixed partitioning logic for equal client distribution
- **Type Safety**: Proper type hints and imports for Flower framework

### 🔧 Issues to Address
- **Typo in filename**: `cleint.py` should be `client.py` (cosmetic issue, functionality works)
- **Model architecture bug**: Line 12 in `model.py` has incorrect input size (128 should be 120)

### 🚀 Ready for Next Steps
- **Federated Training Loop**: Complete the simulation start in `main.py` (currently strategy is defined but not started)
- **Experiment Tracking**: Enhanced logging and result visualization
- **Model Architecture Fix**: Correct the input size in `model.py` line 12

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

## Configuration Parameters
- `num_rounds: 10` - Federated learning rounds
- `num_clients: 100` - Number of participating clients
- `batch_size: 20` - Training batch size
- `num_classes: 10` - MNIST digit classes (0-9)
- `num_clients_per_round_fit: 10` - Minimum clients selected for training per round
- `num_clients_per_round_eval: 25` - Minimum clients selected for evaluation per round
- `config_fit` - Local training parameters:
  - `lr: 0.01` - Learning rate
  - `momentum: 0.9` - SGD momentum
  - `local_epochs: 1` - Local training epochs

## Next Steps
1. **Complete federated training loop** in `main.py` - Add `fl.simulation.start_simulation()` call
2. **Fix model architecture** - Correct input size in `model.py` line 12 (128 → 120)
3. **Rename file** from `cleint.py` to `client.py` (optional cosmetic change)
4. **Add experiment tracking** and result visualization
5. **Test end-to-end** federated learning workflow

## Technical Notes
- **Data Partitioning**: Currently uses IID (Independent and Identically Distributed) partitioning
- **Model Architecture**: CNN suitable for 28x28 grayscale images
- **Client Training**: Each client trains locally with SGD optimizer
- **Parameter Aggregation**: Ready for FedAvg algorithm implementation
