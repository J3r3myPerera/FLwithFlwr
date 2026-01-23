# Simplified Stratified Client Selection for Federated Learning

## Overview

This is a **simplified version** of the Federated Learning implementation that focuses **exclusively on Stratified Client Selection**. All adaptive mu and multi-layer mu functionality has been removed to provide a clean, focused implementation.

## What's Included

### Core Files (Simplified)

1. **`main_stratified.py`** - Main entry point (simplified)
   - Only `compare_stratified` and `stratified` strategies
   - No adaptive mu or multi-layer mu code
   - Clean, focused implementation

2. **`client_stratified.py`** - Flower client (simplified)
   - Basic FedProx support (single mu value)
   - No layer-specific mu
   - No divergence computation

3. **`model_stratified.py`** - Neural network model (simplified)
   - Basic FedProx proximal term
   - No layer-specific mu support
   - Clean training loop

4. **`server_stratified.py`** - Server configuration (simplified)
   - Basic fit configuration
   - No adaptive mu functions
   - Simple evaluation function

5. **`conf/stratified.yaml`** - Simplified configuration
   - Only essential parameters
   - No adaptive/multi-layer mu settings
   - Clean, minimal config

### Stratified Selection Files (Unchanged)

6. **`stratified_selector.py`** - Core stratified selection logic
7. **`stratified_strategy.py`** - Flower strategy with stratified selection (cleaned up)
8. **`dataset.py`** - Data loading and partitioning
9. **`plotting.py`** - Visualization utilities

## Quick Start

### Using Simplified Version

```bash
cd /Users/dinukaperera/FLwithFlwr/flowertry

# Run with simplified main script
python main_stratified.py --config-name=stratified
```

### Configuration

Edit `conf/stratified.yaml`:

```yaml
# Strategy
strategy: compare_stratified  # or 'stratified'

# Basic settings
num_rounds: 20
num_clients: 12
batch_size: 32
num_clients_per_round_fit: 6
alpha: 0.5  # Data heterogeneity

# FedProx (simple, single mu)
fedprox:
  mu: 0.01  # Single mu value for all layers
  lr: 0.001

# Stratified selection
min_clients_per_stratum: 1
```

## Key Differences from Full Version

| Feature | Full Version | Simplified Version |
|---------|-------------|-------------------|
| **Strategies** | FedAvg, FedProx, Compare, Compare Adaptive, Compare Stratified | Compare Stratified, Stratified only |
| **Mu Configuration** | Single mu, Layer-specific mu, Adaptive mu | Single mu only |
| **Client Code** | Divergence computation, Layer-specific mu | Basic FedProx only |
| **Model Training** | Layer-specific proximal terms | Single proximal term |
| **Server Functions** | Adaptive mu config, Layer mu config | Basic config only |
| **Configuration Files** | base.yaml (complex) | stratified.yaml (simple) |

## File Mapping

If you want to use the simplified version, replace these imports:

```python
# OLD (full version)
from client import generate_client_fn
from model import DisposableIncomeModel, train, test
from server import get_on_fit_config, get_evaluate_fn
import main

# NEW (simplified version)
from client_stratified import generate_client_fn
from model_stratified import DisposableIncomeModel, train, test
from server_stratified import get_on_fit_config, get_evaluate_fn
import main_stratified
```

## What Was Removed

### 1. Adaptive Mu Controller
- `adaptive_mu.py` - Not needed
- `AdaptiveMuController` class
- `AdaptiveMuConfig` dataclass
- Divergence-based mu adaptation
- Schedule-based mu decay

### 2. Multi-Layer Mu
- Layer-specific mu values in config
- `layer_mus` parameter in training
- Layer name mapping functions
- Per-layer proximal terms

### 3. Divergence Computation
- `compute_layer_divergences()` function
- Divergence metrics in client fit
- Divergence aggregation in strategy
- Divergence history tracking

### 4. Complex Strategy Modes
- `compare` - Base vs Multi-Layer FedProx
- `compare_adaptive` - 3-way comparison
- `fedavg` - Standard FedAvg
- `fedprox` - Standard FedProx

### 5. Adaptive Plotting
- `plot_adaptive_mu_evolution()`
- Divergence evolution plots
- Mu schedule factor plots

## Advantages of Simplified Version

✅ **Cleaner Code**
- Easier to understand
- Fewer dependencies
- Less complexity

✅ **Faster Execution**
- No divergence computation overhead
- Simpler training loop
- Reduced memory usage

✅ **Focused on Research**
- Pure stratified selection
- Clear comparison with random
- Easier to explain in thesis

✅ **Easier to Modify**
- Less code to navigate
- Clear structure
- Simple configuration

## Running Experiments

### 1. Compare Random vs Stratified

```bash
python main_stratified.py --config-name=stratified strategy=compare_stratified
```

This runs:
- Phase 1: Random client selection (baseline)
- Phase 2: Stratified client selection
- Generates comparison plots

### 2. Run Only Stratified

```bash
python main_stratified.py --config-name=stratified strategy=stratified
```

This runs:
- Only stratified selection
- No random baseline
- Generates stratified analysis plots

### 3. Adjust Parameters

```bash
# High heterogeneity
python main_stratified.py --config-name=stratified alpha=0.1

# More clients per round
python main_stratified.py --config-name=stratified num_clients_per_round_fit=9

# Different mu value
python main_stratified.py --config-name=stratified fedprox.mu=0.05

# More rounds
python main_stratified.py --config-name=stratified num_rounds=50
```

## Output Structure

```
outputs/
└── <date>/
    └── <time>/
        ├── stratified_selection_analysis.png
        ├── random_vs_stratified_comparison.png
        ├── comparison_plot.png
        ├── summary_comparison.png
        ├── results.pkl
        └── .hydra/
            └── config.yaml
```

## Configuration Reference

### Essential Parameters

```yaml
# Strategy
strategy: compare_stratified  # compare_stratified or stratified

# FL Settings
num_rounds: 20                 # Number of training rounds
num_clients: 12                # Total clients (must be divisible by 3 for tiers)
batch_size: 32                 # Batch size for training
num_clients_per_round_fit: 6   # Clients selected per round
num_clients_per_round_eval: 6  # Clients for evaluation

# Data
alpha: 0.5                     # Heterogeneity (0.1=high, 0.5=moderate, 1.0=low)
non_iid: true                  # Use Non-IID partitioning

# Training
config_fit:
  lr: 0.001                    # Learning rate
  momentum: 0.0                # SGD momentum
  local_epochs: 5              # Local training epochs

# FedProx
fedprox:
  mu: 0.01                     # Proximal term (0.0=FedAvg, >0=FedProx)
  lr: 0.001                    # Learning rate override

# Stratified Selection
min_clients_per_stratum: 1     # Minimum per City Tier (fairness)
```

## Comparison: Full vs Simplified

### Full Version (base.yaml)
```yaml
strategy: compare_adaptive

fedprox_base:
  mu: 0.1
  
fedprox_multilayer:
  mu: 0.01
  layer_mus:
    input: 0.25
    hidden1: 0.30
    hidden2: 0.35
    output: 0.40

fedprox_adaptive:
  schedule_type: cosine
  min_schedule_factor: 0.2
  use_divergence: true
  divergence_weight: 0.6
```

### Simplified Version (stratified.yaml)
```yaml
strategy: compare_stratified

fedprox:
  mu: 0.01  # Single value, simple
```

## Migration Guide

### From Full to Simplified

1. **Update imports**:
   ```python
   # Change
   from client import generate_client_fn
   # To
   from client_stratified import generate_client_fn
   ```

2. **Update config**:
   ```bash
   # Change
   python main.py strategy=compare_stratified
   # To
   python main_stratified.py --config-name=stratified
   ```

3. **Remove unused configs**:
   - Delete `fedprox_base`, `fedprox_multilayer`, `fedprox_adaptive` sections
   - Keep only `fedprox` with single `mu` value

### From Simplified to Full

If you need advanced features later:

1. Use original files: `main.py`, `client.py`, `model.py`, `server.py`
2. Use `conf/base.yaml` configuration
3. Add back adaptive mu imports if needed

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure you're using the simplified files:
```python
from client_stratified import generate_client_fn
from model_stratified import DisposableIncomeModel
from server_stratified import get_on_fit_config
```

### Issue: Config not found

**Solution**: Use `--config-name=stratified`:
```bash
python main_stratified.py --config-name=stratified
```

### Issue: Missing layer_mus

**Solution**: The simplified version doesn't use `layer_mus`. Remove it from config or use single `mu` value.

## Documentation

- **Full Documentation**: `STRATIFIED_SELECTION_README.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `ARCHITECTURE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **This File**: `README_SIMPLIFIED.md`

## Recommended Usage

### For Thesis/Research

Use the **simplified version** if:
- ✅ You only need stratified client selection
- ✅ You want cleaner, easier-to-explain code
- ✅ You don't need adaptive or layer-specific mu
- ✅ You want faster experiments

Use the **full version** if:
- ❌ You need adaptive mu functionality
- ❌ You want to compare multiple strategies
- ❌ You need layer-specific regularization
- ❌ You want all advanced features

### For Production

The simplified version is recommended for production deployments:
- Less complexity = fewer bugs
- Easier to maintain
- Faster execution
- Clearer code for team members

## Summary

The simplified version provides:
- ✅ Clean stratified client selection
- ✅ Basic FedProx support (single mu)
- ✅ All fairness metrics and visualizations
- ✅ Easy to understand and modify
- ✅ Perfect for thesis research

What's removed:
- ❌ Adaptive mu controller
- ❌ Layer-specific mu values
- ❌ Divergence computation
- ❌ Complex strategy comparisons
- ❌ Adaptive plotting functions

**Result**: A focused, clean implementation of stratified client selection for your personal finance FL research.

---

**Version**: Simplified 1.0  
**Date**: January 23, 2026  
**Purpose**: Clean stratified selection without adaptive/multi-layer complexity
