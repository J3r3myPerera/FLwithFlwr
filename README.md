# Flower Federated Learning Project

## Project Overview
This project is a Federated Learning implementation using the Flower framework. It was initialized on October 24, 2025, as part of learning and experimenting with federated learning concepts.

## Changes Made Today (October 24, 2025)

### Initial Project Setup
- **Created project structure**: Established the basic directory structure for a Flower federated learning project
- **Set up Hydra configuration**: Implemented configuration management using Hydra framework
- **Configured VS Code environment**: Set up Python interpreter path for the `flower_tutorial` conda environment

### Files Created

#### 1. `main.py`
- Main entry point for the federated learning application
- Uses Hydra for configuration management
- Currently displays the configuration using `OmegaConf.to_yaml(cfg)`
- **Updated**: Added dataset preparation call to `prepare_dataset()` function
- Ready for federated learning implementation

#### 2. `dataset.py` (NEW)
- Dataset handling module for MNIST data
- Implements `get_minst()` function for downloading and preparing MNIST dataset
- Includes data preprocessing with normalization transforms
- Implements `prepare_dataset()` function for data partitioning (incomplete)
- Uses PyTorch and Torchvision for data handling

#### 3. `conf/base.yaml`
- Configuration file defining federated learning parameters:
  - `num_rounds: 10` - Number of federated learning rounds
  - `num_clients: 100` - Number of client participants
  - `config_fit` - Training configuration:
    - `lr: 0.01` - Learning rate
    - `momentum: 0.9` - Momentum for optimizer

#### 3. `.vscode/settings.json`
- VS Code workspace configuration
- Points to the `flower_tutorial` conda environment
- Enables automatic environment activation in terminal

#### 4. Output Directory Structure
- Created `outputs/2025-10-24/00-40-15/` directory for experiment results
- Includes Hydra-generated configuration files:
  - `.hydra/config.yaml` - Runtime configuration
  - `.hydra/hydra.yaml` - Hydra framework settings
  - `.hydra/overrides.yaml` - Configuration overrides
- `main.log` - Application log file (currently empty)

## Project Structure
```
flowertry/
├── conf/
│   └── base.yaml          # Main configuration file
├── main.py                # Application entry point
├── dataset.py             # Dataset handling module (NEW)
├── outputs/               # Experiment outputs
│   └── 2025-10-24/       # Date-based organization
│       └── 00-40-15/     # Time-based experiment folder
│           ├── .hydra/    # Hydra configuration files
│           └── main.log   # Application logs
├── .vscode/
│   └── settings.json      # VS Code workspace settings
└── README.md              # This file
```

## Recent Updates

### Dataset Implementation (Latest)
- **Added `dataset.py`**: New module for handling MNIST dataset
- **MNIST Integration**: Implemented data loading with proper preprocessing
- **Data Preprocessing**: Added normalization transforms (mean: 0.1307, std: 0.3081)
- **Data Partitioning**: Started implementation of `prepare_dataset()` for federated learning
- **Integration**: Connected dataset preparation to main application flow

### Current Status
- Dataset module is partially implemented (function signature exists but implementation incomplete)
- Main application now calls dataset preparation function
- Ready for completing the data partitioning logic for federated learning

## Environment Setup
- **Python Environment**: `flower_tutorial` conda environment
- **Framework**: Flower (Federated Learning)
- **Configuration**: Hydra
- **Data Processing**: PyTorch, Torchvision
- **IDE**: VS Code with Python extension

## Next Steps
The project is progressing well with dataset handling now implemented. Immediate next steps include:
- **Complete `prepare_dataset()` function**: Finish the data partitioning logic for federated learning
- **Fix import issues**: Add missing imports in `dataset.py` (Compose, Normalize)
- **Implement client and server logic**: Create Flower client and server components
- **Add model definitions**: Define the neural network architecture
- **Implement federated learning algorithms**: Set up the federated training loop
- **Add evaluation and monitoring capabilities**: Implement metrics and logging

## Running the Project
To run the current setup:
```bash
python main.py
```

This will execute the main function with the default configuration from `conf/base.yaml` and display the configuration parameters.

## Configuration
The project uses Hydra for configuration management, allowing for:
- Easy parameter modification
- Experiment tracking
- Configuration versioning
- Command-line overrides

Example of running with custom parameters:
```bash
python main.py num_rounds=20 num_clients=50
```
