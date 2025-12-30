# Federated Learning for Personal Finance Modeling

## Project Overview
This project implements **Federated Learning** using the Flower framework. The initial use-case is **Savings Potential Classification** based on the Indian Personal Finance and Spending Habits dataset, but the current implementation and recent updates use the MNIST dataset for federated experiments and bug testing.

## Recent Changes and Implementation Status

### Latest Fixes and Enhancements (as of December 2025)
- **Bug Fixes:**
  - Fixed missing import of the `test` function in `server.py`.
  - Corrected a critical dimension mismatch in `fc2` layer of the MNIST CNN model.
  - Fixed assignment and method call bugs in `cleint.py` (`set_parameters`, assignments, type hints).
  - Resolved data partitioning issues in `dataset.py`.

- **Enhancements:**
  - Added `client_resources` parameter for optimal CPU/GPU allocation (especially for Apple Silicon).
  - Added automatic result saving (`results.pkl`) for experiment tracking.
  - Integrated a configurable Flower FedAvg server-side strategy.
  - Hydra config updated (`conf/base.yaml`) to include new simulation hyperparameters.

- **Project Structure Overhaul:**  
  All major Flower experiment code is now organized under `flowertry/`:
    - `main.py`
    - `model.py` (CNN for MNIST, original MLP for Indian Finance can be restored if desired)
    - `cleint.py` (client implementation, typo intentionally retained for now)
    - `server.py` (server config and evaluation)
    - `dataset.py` (MNIST partitioning)
    - `conf/base.yaml` (config)
    - `outputs/` (experiment results)

### Current Features
- **Data Pipelining**: MNIST loading, preprocessing, and IID partitioning.
- **Model Architectures**:
  - **For MNIST**: Convolutional Neural Network (CNN), fully debugged.
  - **For Indian Finance**: (Prior version: Multi-layer MLP, see below for details).
- **Federated Client Implementation**: All core Flower protocol methods fully implemented and bug-fixed.
- **Server Strategy**: Configurable FedAvg, hyperparameter round-wise configuration, log/tracking support.
- **Result Persistence**: All experiment histories are saved for easy analysis.
- **Resource Configuration**: Supports CPU and Apple M1 chip settings.
- **Configuration Management**: All settings now centralized with Hydra.

---

## How to Run

From the project root:
```bash
cd flowertry
python main.py
```

---

## Datasets and Task Variants

### 1. Indian Personal Finance and Spending Habits Dataset (Planned/Future)

**Objective:** Classify users into savings potential categories based on their financial profiles.

**Target variable**:  
`Desired_Savings_Percentage` (binned into 3 classes):
- **Low (Class 0):** < 7% savings rate
- **Medium (Class 1):** 7-12% savings rate  
- **High (Class 2):** > 12% savings rate

**Features:**  
Total 16 (demographics, spending categories, occupation, etc.)

**Proposed MLP Architecture:**
- Input: 16 features
- Hidden 1: 64 (ReLU, BatchNorm, Dropout)
- Hidden 2: 32 (ReLU, BatchNorm, Dropout)
- Hidden 3: 16 (ReLU, BatchNorm)
- Output: 3-class (CrossEntropyLoss)

---

### 2. MNIST Dataset (Current, Debugged Implementation)

**Model:**  
- CNN: 2 Conv layers + 2 Pooling + 3 Fully Connected layers
- Final bug: fixed FC2 to input size 120 (from 128)

**Federated Pipeline:**  
- IID partitioning among clients  
- Result saving and full experiment workflow

---

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

---

## Notes & Environment

- Typo: `cleint.py` should be `client.py` (to be fixed).
- Uses Python, Flower, PyTorch, Torchvision, Hydra.
- Tested on M1 MacBook Pro, workflow may require adjustment for non-M1 hardware.

---

## Future Steps

- Restore and productionize the Indian Personal Finance MLP/classification pipeline if experimental phase with MNIST is complete.
- Add scripts for results analysis and visualization.
- Switch `cleint.py` to `client.py` once all references are updated.

---
```bash
cd flowertry
python main.py
```

## Project Structure

Depending on the active experiment, the directory structure and dataset will be slightly different.

### For Indian Personal Finance & Spending Habits Experiment:
```
FLwithFlwr/
├── README.md
└── flowertry/
    ├── conf/
    │   └── base.yaml                # Configuration file
    ├── data/
    │   └── indianPersonalFinanceAndSpendingHabits/
    │       └── indianPersonalFinanceAndSpendingHabits.csv
    ├── main.py                      # Main entry point
    ├── dataset.py                   # Data loading and preprocessing
    ├── model.py                     # MLP neural network
    ├── cleint.py                    # Flower client implementation (to be renamed client.py)
    ├── server.py                    # Server-side evaluation and config
    └── outputs/                     # Experiment results (organized by date)
```

### For MNIST Experiment:
- The data folder and CSV above are replaced with automatic downloads of the MNIST dataset.
- Model is a CNN for digit recognition.
- Parameters for MNIST are shown below.

---

## Configuration Parameters

Depending on the experiment, key config is specified in `base.yaml`. Typical main options:

**For Indian Personal Finance experiment:**
```yaml
# Federated Learning Settings
num_rounds: 10           # Number of FL rounds
num_clients: 10          # Total number of clients
batch_size: 32           # Batch size for training

# Client Selection
num_clients_per_round_fit: 5    # Clients for training per round
num_clients_per_round_eval: 10  # Clients for evaluation per round

# Local Training
config_fit:
  lr: 0.01              # Learning rate
  momentum: 0.9         # SGD momentum
  local_epochs: 3       # Local training epochs
```

**For MNIST experiment:**
```yaml
num_rounds: 10
num_clients: 100
batch_size: 20
num_classes: 10
num_clients_per_round_fit: 10
num_clients_per_round_eval: 25
config_fit:
  lr: 0.01
  momentum: 0.9
  local_epochs: 1
```

---

## Dataset

### Indian Personal Finance and Spending Habits
- **Samples:** 20,000 records
- **Features:** 16 preprocessed (14 numerical + 2 categorical encoded)
- **Target:** 3-class classification (Low/Medium/High savings potential)
- **Preprocessing Steps**:
  1. StandardScaler for numericals
  2. LabelEncoder for categoricals (Occupation, City_Tier)
  3. Target: <7% → Low, 7-12% → Medium, >12% → High
  4. 80%/10%/10% train/val/test split
  5. IID partitioning across federated clients

### MNIST
- Standard torchvision MNIST, 60,000 train + 10,000 test images, 10 classes.
- Partitioned IID across clients.

---

## Federated Learning Setup

Uses **FedAvg** (Federated Averaging):

- Aggregates model parameters from selected clients each round.
- Server evaluates global model on test/validation set every round.
- Accuracy progression tracked per round.

#### Client Training
- Each client trains locally for `local_epochs`.
- SGD optimizer with momentum.
- Updated weights sent to server after each round.

---

## Example Workflow Steps

1. Load and display config parameters.
2. Prepare and partition dataset (Indian Finance CSV or MNIST).
3. Generate client functions for federated learning.
4. Display basic dataset statistics.
5. Start federated simulation (FedAvg, selection as per config).
6. Execute training rounds with subset client selection per fit/eval.
7. Evaluate and log accuracy/progression each round.
8. Save training history (pickle) in output directory via Hydra.

---

## Expected Output (Indian Personal Finance, 3-Class)
```
============================================================
FEDERATED LEARNING - SAVINGS POTENTIAL CLASSIFICATION
============================================================

Class Distribution:
  Class 0 (Low <7%): ~6,500 samples (32.5%)
  Class 1 (Medium 7-12%): ~7,000 samples (35.0%)
  Class 2 (High >12%): ~6,500 samples (32.5%)

Accuracy progression:
  Round 0: ~35% (random)
  Round 5: ~55-60%
  Round 10: ~65-70%
```

## Expected Output (MNIST)
- Accuracy typically grows from ~10% (random) to 75-90%, depending on config and rounds.

---

## Environment Setup

```bash
# For both experiments:
conda create -n flower_tutorial python=3.10
conda activate flower_tutorial
# Indian Finance:
pip install flwr torch pandas scikit-learn hydra-core omegaconf
# For MNIST:
pip install flwr torch torchvision hydra-core omegaconf
```

---

## Technical Notes

- **Framework**: Flower (FedAvg strategy)
- **Deep Learning**: PyTorch (MLP for tabular, CNN for MNIST)
- **Configuration**: Hydra
- **Data Processing**: pandas, scikit-learn, torchvision
- **Partitioning**: Current implementations are IID; future work will include Non-IID.
- **Hardware**: Optimized for M1 MacBook Pro (CPU; MPS/CUDA disabled for Ray compatibility)
- **Result Storage**: Hydra-managed outputs; results.pkl file for accuracy & history
- **GPU**: No explicit GPU by default on M1, but PyTorch MPS supported for local (non-Ray) runs.

---

## Future Next Steps & Enhancements

1. **Data Partitioning**: Add Dirichlet-based non-IID support.
2. **Aggregation**: Implement FedProx/FedAvgM, support for client heterogeneity.
3. **Metrics**: Add F1, per-class precision/recall.
4. **Tuning**: Grid-search for best MLP/CNN architecture.
5. **Result Analysis Scripts**: For automatic analysis of `results.pkl`.
6. **Logging/Tracking**: Enhanced logging & experiment tracking.
7. **File Rename**: Once stable, change `cleint.py` to `client.py`.
8. **Performance Optimization**: Tune `client_resources`, batch sizes per hardware.
9. **Model Evaluation**: Add visualization, more robust checks.
10. **Experiment Documentation**: Update project as new experiments are added.
