# Federated Learning for Personal Finance Modeling

## Project Overview
This project implements **Federated Learning** using the Flower framework for **Savings Potential Classification** based on the Indian Personal Finance and Spending Habits dataset.

## Task: Savings Potential Classification

**Objective:** Classify users into savings potential categories based on their financial profile.

### Target Variable
`Desired_Savings_Percentage` discretized into 3 classes:
- **Low (Class 0):** < 7% savings rate
- **Medium (Class 1):** 7-12% savings rate  
- **High (Class 2):** > 12% savings rate

### Input Features (16 total)
- **Demographics:** Income, Age, Dependents
- **Location:** City_Tier (encoded)
- **Occupation:** Occupation type (encoded)
- **Spending Categories:** Rent, Loan_Repayment, Insurance, Groceries, Transport, Eating_Out, Entertainment, Utilities, Healthcare, Education, Miscellaneous

### Model Architecture
**Multi-layer Neural Network (MLP):**
- Input Layer: 16 features
- Hidden Layer 1: 64 neurons (ReLU, BatchNorm, Dropout)
- Hidden Layer 2: 32 neurons (ReLU, BatchNorm, Dropout)
- Hidden Layer 3: 16 neurons (ReLU, BatchNorm)
- Output Layer: 3 classes (softmax via CrossEntropyLoss)

### Evaluation Metric
- **Accuracy** (primary metric for classification)

## Quick Start

```bash
cd flowertry
python main.py
```

## Project Structure
```
FLwithFlwr/
├── README.md                    # This file
└── flowertry/
    ├── conf/
    │   └── base.yaml            # Configuration file
    ├── data/
    │   └── indianPersonalFinanceAndSpendingHabits/
    │       └── indianPersonalFinanceAndSpendingHabits.csv
    ├── main.py                  # Main entry point
    ├── dataset.py               # Data loading and preprocessing
    ├── model.py                 # MLP neural network
    ├── cleint.py                # Flower client implementation
    ├── server.py                # Server-side evaluation
    └── outputs/                 # Experiment results
```

## Configuration Parameters

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

## Dataset

**Indian Personal Finance and Spending Habits Dataset**
- **Samples:** 20,000 records
- **Features:** 16 preprocessed features (14 numerical + 2 categorical encoded)
- **Target:** 3-class classification (Low/Medium/High savings potential)

### Data Preprocessing
1. Numerical features: StandardScaler normalization
2. Categorical features: LabelEncoder (Occupation, City_Tier)
3. Target discretization: <7% → Low, 7-12% → Medium, >12% → High
4. Train/Val/Test split: 80%/10%/10%
5. IID partitioning across federated clients

## Federated Learning Setup

### Strategy: FedAvg (Federated Averaging)
- Aggregates model parameters from selected clients each round
- Server-side evaluation on global test set
- Tracks accuracy progression across rounds

### Client Training
- Each client trains locally for `local_epochs` epochs
- Uses SGD optimizer with momentum
- Sends updated parameters to server

## Expected Output

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

## Environment Setup

```bash
# Create conda environment
conda create -n flower_tutorial python=3.10
conda activate flower_tutorial

# Install dependencies
pip install flwr torch pandas scikit-learn hydra-core omegaconf
```

## Technical Notes

- **Framework:** Flower (Federated Learning)
- **Deep Learning:** PyTorch
- **Configuration:** Hydra
- **Data Processing:** pandas, scikit-learn
- **Hardware:** Optimized for CPU (MPS/CUDA disabled for Ray compatibility)

## Future Enhancements

1. **Non-IID Data Simulation:** Implement Dirichlet-based partitioning
2. **FedProx:** Add proximal term for heterogeneous data
3. **Additional Metrics:** F1-Score, per-class precision/recall
4. **Hyperparameter Tuning:** Grid search for optimal MLP architecture
