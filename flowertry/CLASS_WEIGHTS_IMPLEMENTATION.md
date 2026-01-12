# Class Weights Implementation

## Overview

Added class weights support to handle imbalanced datasets in federated learning. This is particularly important for the savings classification task where class distribution may be skewed, especially with non-IID data partitioning.

## Implementation Details

### 1. Dataset Module (`dataset.py`)

- **New Function**: `compute_class_weights(labels, method='balanced')`
  - Computes class weights based on label distribution
  - Two methods:
    - `'balanced'`: Inverse frequency weighting → `w_i = n_samples / (n_classes * count_i)`
    - `'sqrt'`: Softer weighting → `w_i = sqrt(n_samples / (n_classes * count_i))`
- **Updated Function**: `prepare_dataset()`
  - Added parameters: `use_class_weights` (bool), `class_weight_method` (str)
  - Returns 4-tuple: `(trainloaders, valloaders, testloader, class_weights)`
  - Computes class weights on full training set before partitioning

### 2. Model Module (`model.py`)

Updated all training and evaluation functions to accept `class_weights` parameter:

- `train()`: Regular FedAvg training with weighted loss
- `train_fedprox()`: FedProx training with weighted loss + proximal term
- `train_scaffold()`: SCAFFOLD training with weighted loss + control variates
- `test()`: Evaluation with weighted loss

All functions create weighted `CrossEntropyLoss` when class_weights provided:

```python
if class_weights is not None:
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
else:
    criterion = nn.CrossEntropyLoss()
```

### 3. Client Module (`cleint.py`)

- **FlowerClient**:
  - Stores `class_weights` in `__init__`
  - Passes to all training/evaluation calls
- **ScaffoldClient**: Inherits class_weights from FlowerClient
- **generate_client_fn()**: Accepts and passes class_weights to client creation

### 4. Server Module (`server.py`)

- **get_evaluate_fn()**:
  - Accepts `class_weights` parameter
  - Passes to `test()` function for server-side evaluation

### 5. Strategy Comparison (`compare_strategies.py`)

- **All strategy runners** (`run_fedavg`, `run_fedprox`, `run_fedscaffold`):

  - Accept `class_weights` parameter
  - Pass to `get_evaluate_fn()` for server-side evaluation
  - SCAFFOLD creates client_fn with class_weights

- **Main function**:
  - Reads config: `use_class_weights` and `class_weight_method`
  - Calls `prepare_dataset()` to compute class_weights
  - Passes class_weights to `generate_client_fn()`
  - Passes class_weights to all strategy runners

### 6. Configuration Files

Added to both `conf/base.yaml` and `conf/baseSoFar.yaml`:

```yaml
# Class weighting for imbalanced datasets
use_class_weights: true # Whether to use class weights in loss function
class_weight_method: "balanced" # 'balanced' (inverse frequency) or 'sqrt' (softer weighting)
```

**Default settings:**

- `base.yaml`: `use_class_weights: false` (baseline comparison)
- `baseSoFar.yaml`: `use_class_weights: true` (optimized configuration)

## Why Class Weights Matter

### Problem: Imbalanced Classes in Non-IID FL

1. **Dataset-level imbalance**: Original dataset may have unequal class distribution
2. **Dirichlet partitioning**: Non-IID split (alpha=0.1) exacerbates imbalance at client level
3. **Minority class underlearning**: Model biased toward majority classes

### Solution: Weighted Loss Function

- Penalizes misclassifications of minority classes more heavily
- Formula: `w_i = n_samples / (n_classes * count_i)`
  - Rare class (count=100) → higher weight (e.g., 10.0)
  - Common class (count=1000) → lower weight (e.g., 1.0)

### Expected Impact

- Improved recall on minority classes
- Better balanced accuracy across all classes
- More robust to extreme non-IID conditions

## Usage

### Enable Class Weights

```yaml
# In conf/baseSoFar.yaml
use_class_weights: true
class_weight_method: "balanced" # or "sqrt" for softer weighting
```

### Run Comparison

```bash
python compare_strategies.py --config-name baseSoFar
```

### Disable Class Weights (Baseline)

```yaml
use_class_weights: false
```

## Code Flow

```
1. main() reads config (use_class_weights, class_weight_method)
2. prepare_dataset() computes class_weights from full training set
3. class_weights passed to generate_client_fn()
4. FlowerClient stores class_weights in __init__
5. Client.fit() passes class_weights to train/train_fedprox/train_scaffold
6. Training functions create weighted CrossEntropyLoss
7. Strategy runners pass class_weights to get_evaluate_fn()
8. Server evaluation uses weighted loss for consistency
```

## Implementation Status

✅ **Complete** - All components updated and tested

- Dataset computation: ✅
- Model training functions: ✅
- Client classes: ✅
- Server evaluation: ✅
- Strategy runners: ✅
- Configuration files: ✅
- Main function integration: ✅

## Next Steps

1. Run experiments with `use_class_weights=true` on baseSoFar config
2. Compare results with baseline (use_class_weights=false)
3. Analyze per-class metrics to verify improved minority class performance
4. Consider tuning class_weight_method ('balanced' vs 'sqrt')
