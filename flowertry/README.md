# Federated Learning for Disposable Income Regression

A comprehensive implementation of federated learning strategies for disposable income prediction using Flower framework with Hydra configuration management.

## Features

- **Multiple FL Strategies**: FedAvg, FedProx, SCAFFOLD, and Hybrid (FedProx + SCAFFOLD)
- **Adaptive Hybrid**: Enhanced hybrid approach with adaptive regularization, warmup, and learning rate scheduling
- **Real-time Visualization**: Live metrics plotting during training
- **Comprehensive Analysis**: Automatic generation of comparison plots and CSV exports
- **Hydra Configuration**: Easy experiment management with organized output directories

## Project Structure

```
flowertry/
├── main.py                  # Main entry point with Hydra
├── client.py                # FL client implementation
├── server.py                # FL server strategies
├── model.py                 # Neural network architecture
├── dataset.py               # Dataset preparation
├── visualize_metrics.py     # Visualization module
├── compare_strategies.py    # Strategy comparison script
├── conf/
│   └── base.yaml           # Configuration file
├── data/
│   └── IndianPersoalFinance/
│       └── indianPersonalFinanceAndSpendingHabits.csv
└── outputs/                 # Hydra managed outputs
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── comparison_results.json
            ├── metrics_history.csv
            └── *.png (plots)
```

## Installation

```bash
# Create conda environment
conda create -n flower_fl python=3.10
conda activate flower_fl

# Install dependencies
pip install flwr==1.20.0 torch torchvision numpy pandas scikit-learn matplotlib hydra-core omegaconf
```

## Usage

### Single Strategy Run

```bash
# Run with default config (hybrid)
python main.py

# Run specific strategy
python main.py strategy=fedavg
python main.py strategy=fedprox
python main.py strategy=scaffold
python main.py strategy=hybrid
```

### Compare Multiple Strategies

```bash
# Compare all strategies
python main.py compare_all=true

# Compare specific strategies (edit conf/base.yaml)
# Set: strategies: [fedavg, hybrid]
python main.py compare_all=true

# Or use comparison script
python compare_strategies.py
```

### Hyperparameter Tuning

```bash
# Tune hybrid weights
python main.py strategy=hybrid hybrid.fedprox_weight=0.3 hybrid.scaffold_weight=0.5

# Tune FedProx mu
python main.py strategy=fedprox fedprox.mu=0.2

# Multiple parameters
python main.py strategy=hybrid hybrid.mu=0.08 hybrid.adaptive_mu=true num_rounds=100
```

## Configuration

Edit `conf/base.yaml`:

```yaml
# Strategy selection
strategy: hybrid

# Training parameters
num_rounds: 50
num_clients: 3
local_epochs: 5
batch_size: 32
learning_rate: 0.01

# Strategy-specific parameters
fedprox:
  mu: 0.1

scaffold:
  learning_rate_correction: 1.0

hybrid:
  fedprox_weight: 0.3
  scaffold_weight: 0.5
  mu: 0.08
  adaptive_mu: true
  mu_decay: 0.98
  warmup_rounds: 5
  warmup_factor: 0.3
  use_lr_scheduler: true

# Visualization
plotting:
  enabled: true
  show_plot: true
  save_plot: true
```

## Strategies

### FedAvg (Baseline)

Standard federated averaging without additional regularization.

### FedProx

Adds proximal term to keep local updates close to global model:

- Good for moderately non-IID data
- Controlled by `mu` parameter

### SCAFFOLD

Uses control variates to reduce client drift:

- Better for heterogeneous data distributions
- Requires more clients (5+) for best results

### Hybrid (Enhanced)

Combines FedProx and SCAFFOLD with advanced features:

- **Adaptive Mu**: Gradually reduces regularization as model converges
- **Warmup**: Lighter regularization in early rounds
- **LR Scheduling**: Learning rate decay for fine-tuning
- **Configurable Weights**: Balance FedProx vs SCAFFOLD contributions

## Output

All results are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:

- `comparison_results.json` - Full metrics for all strategies
- `metrics_history.csv` - Round-by-round metrics
- `*_metrics_plot.png` - Real-time training plots
- `training_history_*.png` - Training curves
- `final_comparison_*.png` - Bar charts of final metrics
- `convergence_analysis_*.png` - Convergence analysis
- `.hydra/` - Configuration snapshots

## Results Interpretation

Lower is better for:

- Loss
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Higher is better for:

- R² Score (coefficient of determination)

## Tips for Best Performance

1. **For your dataset (3 clients, City_Tier split)**:

   - FedProx often performs best
   - Hybrid with reduced weights (0.3/0.5) can match or exceed FedAvg

2. **Hyperparameter Tuning**:

   - Start with FedProx mu=0.1
   - For hybrid, use lighter regularization than pure strategies
   - Enable adaptive_mu for automatic tuning

3. **More clients (5+)**:

   - SCAFFOLD becomes more effective
   - Hybrid shows clearer advantages

4. **Highly non-IID data**:
   - Increase FedProx mu (0.2-0.5)
   - Use hybrid with balanced weights

## Troubleshooting

**Hybrid underperforming?**

- Reduce weights (try 0.3/0.5)
- Enable adaptive_mu
- Increase warmup_rounds

**Training unstable?**

- Lower learning rate
- Increase local_epochs
- Adjust batch_size

**Out of memory?**

- Reduce batch_size
- Disable show_plot (set to false)

## Citation

Built with [Flower](https://flower.dev/) framework and [Hydra](https://hydra.cc/) for configuration management.

## License

Educational/Research Use
