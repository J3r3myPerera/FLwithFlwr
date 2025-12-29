# Flower Federated Learning Project

## Project Overview
This project is a complete Federated Learning implementation using the Flower framework with MNIST dataset. It includes client-server architecture, neural network model, and data partitioning for federated learning experiments.

## Recent Changes (Latest Updates)

### Latest Fixes and Enhancements (Post-Last Git Push - December 2025)
- **Fixed `server.py`**: Added missing `test` import from `model` module (resolved `NameError: name 'test' is not defined`)
- **Fixed `model.py`**: Corrected critical dimension mismatch in `fc2` layer (changed from `nn.Linear(128, 84)` to `nn.Linear(120, 84)`) - this was causing runtime errors during model evaluation
- **Enhanced `main.py`**: 
  - Added `client_resources` parameter to `start_simulation()` with optimized values for M1 Pro MacBook (`num_cpus: 0.8, num_gpus: 0`)
  - Added result saving functionality - saves training history to `results.pkl` file in output directory
- **Fixed `cleint.py`**: Resolved `NameError` by adding missing type imports (`Dict`, `Scalar`, `NDArrays`)
- **Fixed `cleint.py`**: Corrected assignment operator (`==` → `=`) for model initialization
- **Fixed `cleint.py`**: Corrected method calls (`get_parameters` → `set_parameters`) in `fit()` and `evaluate()` methods
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
FLwithFlwr/
├── README.md              # This file
└── flowertry/             # Main project directory
    ├── conf/
    │   └── base.yaml      # Configuration file
    ├── main.py            # Main application entry point
    ├── dataset.py         # Dataset handling and partitioning
    ├── model.py           # Neural network architecture
    ├── cleint.py          # Flower client implementation
    ├── server.py          # Server configuration functions
    ├── outputs/           # Experiment outputs
    │   └── YYYY-MM-DD/    # Date-based organization
    │       └── HH-MM-SS/  # Time-based experiment folder
    │           ├── .hydra/    # Hydra configuration files
    │           ├── main.log   # Application logs
    │           └── results.pkl # Saved training history
    └── .vscode/
        └── settings.json  # VS Code workspace settings
```

## Current Implementation Status

### ✅ Completed Features
- **Data Pipeline**: MNIST loading, preprocessing, and IID partitioning
- **Neural Network**: CNN model for digit classification (architecture bug fixed)
- **Client Implementation**: Complete Flower client with training/evaluation (all bugs fixed)
- **Server Functions**: Server-side configuration and evaluation functions (import issues resolved)
- **Server Strategy**: FedAvg strategy configured in main.py
- **Federated Training Loop**: Complete simulation execution with `fl.simulation.start_simulation()`
- **Resource Optimization**: Client resources configured for M1 Pro MacBook (10 CPU cores)
- **Result Persistence**: Training history automatically saved to pickle file
- **Configuration Management**: Hydra-based parameter configuration
- **Data Splitting**: Fixed partitioning logic for equal client distribution
- **Type Safety**: Proper type hints and imports for Flower framework

### 🔧 Issues to Address
- **Typo in filename**: `cleint.py` should be `client.py` (cosmetic issue, functionality works)

### 🚀 Ready for Next Steps
- **Experiment Tracking**: Enhanced logging and result visualization
- **Result Analysis**: Add scripts to load and analyze saved `results.pkl` files
- **Performance Tuning**: Experiment with different `client_resources` values for optimization

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
cd flowertry
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
1. **Result Analysis**: Create scripts to load and visualize saved `results.pkl` files
2. **Rename file** from `cleint.py` to `client.py` (optional cosmetic change)
3. **Experiment Tracking**: Enhanced logging and result visualization
4. **Performance Optimization**: Fine-tune `client_resources` based on system performance
5. **Model Evaluation**: Add more comprehensive evaluation metrics and visualization

## Technical Notes
- **Data Partitioning**: Currently uses IID (Independent and Identically Distributed) partitioning
- **Model Architecture**: CNN suitable for 28x28 grayscale images (fully connected layers: 256→120→84→num_classes)
- **Client Training**: Each client trains locally with SGD optimizer
- **Parameter Aggregation**: FedAvg algorithm implemented and running
- **Resource Management**: Optimized for M1 Pro MacBook with 10 CPU cores - uses 0.8 CPUs per client to leave headroom for system processes
- **Result Storage**: Training history saved as pickle file in Hydra-managed output directory
- **GPU Support**: Currently set to 0 GPUs (M1 Macs use MPS which is handled separately by PyTorch)
