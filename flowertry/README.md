# Federated Learning Strategy Comparison Framework

A comprehensive framework for comparing federated learning strategies (FedAvg, FedProx, and FedSCAFFOLD) on non-IID data distributions using the Flower framework.

## 🎯 Project Overview

This project implements a federated learning system for **Savings Potential Classification** using the Indian Personal Finance and Spending Habits dataset. The task is to classify users into three categories:

- **Low**: <7% savings potential
- **Medium**: 7-12% savings potential
- **High**: >12% savings potential

## ✨ Key Features

### 1. **Multi-Strategy Comparison**

- **FedAvg**: Standard federated averaging algorithm
- **FedProx**: Handles data heterogeneity with proximal term regularization (with adaptive μ)
- **FedSCAFFOLD**: Addresses client-drift using control variates
- **Hybrid FedProx-SCAFFOLD**: Enhanced approach combining both strategies with:
  - Sequential activation (SCAFFOLD warm-up → gradual FedProx introduction)
  - Dual-μ architecture (separate μ for corrected/uncorrected gradients)
  - Conditional drift-based activation per client

### 2. **Non-IID Data Distribution**

- Implements **Dirichlet distribution** for realistic heterogeneous data partitioning
- Configurable heterogeneity levels via `alpha` parameter
- Supports both IID and non-IID data distributions

### 3. **Comprehensive Evaluation**

- Centralized evaluation on global test set
- Per-round accuracy and loss tracking
- Comparison visualization tools
- Results saved in pickle format for analysis

## 📋 Requirements

```bash
pip install torch torchvision
pip install flwr
pip install hydra-core omegaconf
pip install pandas scikit-learn numpy
pip install matplotlib  # Optional, for visualization
```

## 🚀 Quick Start

### 1. Run Strategy Comparison

Compare all configured strategies:

```bash
python compare_strategies.py
```

### 2. Run Single Strategy

Run a single strategy using the main script:

```bash
python main.py
```

### 3. Visualize Results

After running comparisons, visualize the results:

```bash
python visualize_comparison.py outputs/<date>/<time>/comparison_results.pkl
```

## ⚙️ Configuration

All configuration is managed through `conf/base.yaml`. Key parameters:

### Data Distribution

```yaml
iid: false # true for IID, false for non-IID
alpha:
  0.3 # Dirichlet parameter (lower = more heterogeneous)
  # 0.1 = very heterogeneous
  # 0.3 = moderate heterogeneity (recommended)
  # 0.5 = mild heterogeneity
  # 1.0+ = approaches IID
```

### Federated Learning Settings

```yaml
num_rounds: 30 # Number of federated rounds
num_clients: 100 # Total number of clients
num_clients_per_round_fit: 10 # Clients participating per round (training)
num_clients_per_round_eval: 25 # Clients participating per round (evaluation)
batch_size: 32
```

### Strategy-Specific Parameters

```yaml
proximal_mu: 0.25 # FedProx proximal term (0.01-0.5, higher = stronger regularization)
scaffold_lr: 1.0 # FedSCAFFOLD learning rate
```

### Training Configuration

```yaml
config_fit:
  lr: 0.01 # Learning rate
  momentum: 0.9 # SGD momentum
  local_epochs: 3 # Local training epochs per round
```

### Select Strategies to Run

```yaml
strategies:
  - fedavg
  - fedprox
  # - fedscaffold  # Comment out to skip
```

## 📁 Project Structure

```
flowertry/
├── main.py                 # Single strategy execution
├── compare_strategies.py   # Multi-strategy comparison
├── visualize_comparison.py # Results visualization
├── dataset.py             # Data loading and non-IID partitioning
├── model.py               # MLP model definition
├── cleint.py              # Flower client implementation
├── server.py              # Server-side evaluation functions
├── scaffold_strategy.py   # Custom FedSCAFFOLD implementation
├── hybrid_strategy.py     # Enhanced Hybrid FedProx-SCAFFOLD strategy
├── adaptive_fedprox.py    # Adaptive FedProx with dynamic μ
├── conf/
│   └── base.yaml          # Configuration file
├── data/
│   └── IndianPersoalFinance/
│       └── indianPersonalFinanceAndSpendingHabits.csv
├── outputs/               # Experiment results (auto-generated)
├── ENHANCED_HYBRID_RESULTS.md  # Detailed hybrid implementation docs
└── HYBRID_FEDPROX_SCAFFOLD.md  # Hybrid approach documentation
```

## 🔬 Recent Updates

### Enhanced Hybrid FedProx-SCAFFOLD (Latest - January 2026)

- ✅ **Sequential Activation Strategy**: SCAFFOLD warm-up phase (10 rounds) before introducing FedProx
- ✅ **Dual-μ Architecture**: Separate μ values for corrected (0.001) and uncorrected (0.1) gradients
- ✅ **Conditional Drift Detection**: Per-client activation based on direction vs magnitude drift
- ✅ **μ Annealing**: Gradual μ increase (×1.5 every 5 rounds) from 0.001 to max 0.3
- ✅ **Best Performance**: Achieved 66.55% accuracy, outperforming all baseline strategies

### Adaptive FedProx

- ✅ Dynamic μ adjustment based on client drift metrics
- ✅ Automatic μ tuning during training

### Non-IID Data Distribution

- ✅ Implemented Dirichlet-based non-IID data partitioning
- ✅ Configurable heterogeneity via `alpha` parameter
- ✅ Automatic class distribution visualization
- ✅ Support for both IID and non-IID modes

### Strategy Comparison Framework

- ✅ Automated comparison of FedAvg, FedProx, FedSCAFFOLD, and Hybrid
- ✅ Fair comparison using same dataset and configuration
- ✅ Comprehensive metrics tracking and reporting
- ✅ Results saved for post-analysis

### FedSCAFFOLD Implementation

- ✅ Custom FedSCAFFOLD strategy for Flower framework
- ✅ Control variate handling for client-drift correction
- ✅ Client-side SCAFFOLD training support

### Configuration Improvements

- ✅ Optimized hyperparameters for non-IID scenarios
- ✅ Increased rounds (30) for better convergence observation
- ✅ Tuned `proximal_mu` (0.25) for FedProx
- ✅ Moderate heterogeneity (alpha=0.3) for balanced comparison

## 📊 Understanding Results

### Output Files

After running `compare_strategies.py`, you'll find:

- `comparison_results.pkl`: Pickled results with all metrics
- `comparison_plot.png`: Visualization of accuracy/loss progression (if matplotlib available)
- `compare_strategies.log`: Execution log

### Metrics Tracked

- **Accuracy**: Centralized test accuracy per round
- **Loss**: Centralized test loss per round
- **Final Accuracy**: Best accuracy achieved
- **Training Time**: Time taken for each strategy

### Interpreting Results

- **FedAvg**: Fast initial convergence, may plateau with high heterogeneity
- **FedProx**: Slower start but more stable, better final accuracy on non-IID data
- **FedSCAFFOLD**: Best for extreme heterogeneity, handles client-drift effectively
- **Hybrid (Enhanced)**: Best overall performance - combines SCAFFOLD's variance reduction with FedProx's regularization. Uses sequential activation to let SCAFFOLD calibrate before introducing FedProx.

## 🎛️ Advanced Usage

### Override Configuration via Command Line

```bash
# Run with different alpha value
python compare_strategies.py alpha=0.1

# Run only FedAvg and FedProx
python compare_strategies.py 'strategies=[fedavg,fedprox]'

# Change number of rounds
python compare_strategies.py num_rounds=50

# Use IID data
python compare_strategies.py iid=true
```

### Custom Strategy Selection

Edit `conf/base.yaml`:

```yaml
strategies:
  - fedavg # Always runs
  - fedprox # Always runs
  # - fedscaffold  # Commented out, won't run
```

## 🔍 Tuning Guide

For detailed guidance on tuning strategies and understanding performance differences, see:

- `FEDPROX_TUNING.md`: Comprehensive FedProx tuning guide
- `ENHANCED_HYBRID_RESULTS.md`: Detailed Hybrid implementation and results analysis
- `HYBRID_FEDPROX_SCAFFOLD.md`: Theoretical background on the hybrid approach

### Quick Tuning Tips

1. **If FedAvg outperforms FedProx**:

   - Increase `proximal_mu` (try 0.3-0.5)
   - Run more rounds (30-50)
   - Increase `local_epochs` to 5-10

2. **For extreme heterogeneity**:

   - Use `alpha: 0.1` for very heterogeneous data
   - Consider FedSCAFFOLD for best results
   - Increase `proximal_mu` for FedProx

3. **For faster convergence**:
   - Use `alpha: 0.5-1.0` (milder heterogeneity)
   - Reduce `num_rounds` if needed
   - All strategies converge faster

## 📈 Example Output

```
================================================================================
FEDERATED LEARNING STRATEGY COMPARISON
================================================================================

Data Distribution: Non-IID (Dirichlet, alpha=0.3)

================================================================================
COMPARISON SUMMARY
================================================================================

Strategy                      Final Accuracy    Final Loss    Time (s)    Final Mu
------------------------------------------------------------------------------------
FedAvg                        66.40%            155.77        166.95      -
FedProx (Adaptive)            66.10%            162.12        237.08      0.4447
FedSCAFFOLD                   66.50%            143.88        105.69      -
Hybrid (FedProx+SCAFFOLD)     66.55%            216.52        188.06      -

================================================================================
BEST STRATEGY (by accuracy): Hybrid (FedProx+SCAFFOLD) (66.55%)
================================================================================

Hybrid μ Evolution:
  Phase 1 (warmup): μ = 0 for 10 rounds
  Phase 2 start: μ = 0.001
  Phase 2 final: μ = 0.300 (capped at max_mu)
```

## 🐛 Troubleshooting

### Issue: "list indices must be integers or slices, not list"

- **Solution**: Already fixed in latest version. Ensure you're using updated `dataset.py`

### Issue: Strategies running on IID data when configured for non-IID

- **Solution**: Check that `iid: false` in `conf/base.yaml` and parameters are passed correctly

### Issue: FedAvg outperforming FedProx

- **Solution**: See `FEDPROX_TUNING.md` for detailed guidance. Try:
  - Increasing `proximal_mu` to 0.3-0.5
  - Running more rounds (30+)
  - Using moderate heterogeneity (alpha=0.3-0.5)

## 📚 References

- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- **FedProx**: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- **FedSCAFFOLD**: [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)
- **Flower Framework**: [https://flower.ai](https://flower.ai)

## 📝 License

This project is for research and educational purposes.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

---

**Last Updated**: January 15, 2026
**Version**: 3.0 (with Enhanced Hybrid FedProx-SCAFFOLD strategy)
