# Stratified Client Selection - Quick Reference

## 🚀 Quick Start

```bash
cd /Users/dinukaperera/FLwithFlwr/flowertry
python main.py strategy=compare_stratified
```

## 📋 Available Strategies

| Strategy | Description | Command |
|----------|-------------|---------|
| `fedavg` | Standard FedAvg | `python main.py strategy=fedavg` |
| `fedprox` | Standard FedProx | `python main.py strategy=fedprox` |
| `compare` | Base vs Multi-Layer FedProx | `python main.py strategy=compare` |
| `compare_adaptive` | Base vs Static vs Adaptive | `python main.py strategy=compare_adaptive` |
| `compare_stratified` | **Random vs Stratified Selection** | `python main.py strategy=compare_stratified` |

## ⚙️ Key Parameters

### Data Heterogeneity (alpha)

```bash
# High heterogeneity (90% primary tier, 10% mixed)
python main.py strategy=compare_stratified alpha=0.1

# Moderate heterogeneity (70% primary tier, 30% mixed) [Default]
python main.py strategy=compare_stratified alpha=0.3

# Low heterogeneity (50% primary tier, 50% mixed)
python main.py strategy=compare_stratified alpha=0.5
```

### Clients Per Round

```bash
# Minimum (3 strata × 1 min = 3 clients)
python main.py num_clients_per_round_fit=3

# Default (balanced representation)
python main.py num_clients_per_round_fit=6

# More clients (better proportional allocation)
python main.py num_clients_per_round_fit=9
```

### Fairness Guarantee

```bash
# Minimum 1 client per stratum [Default]
python main.py min_clients_per_stratum=1

# Stricter fairness (minimum 2 per stratum)
python main.py min_clients_per_stratum=2
```

### Training Rounds

```bash
# Quick test (10 rounds)
python main.py num_rounds=10

# Default (20 rounds)
python main.py num_rounds=20

# Extended training (50 rounds)
python main.py num_rounds=50
```

## 📊 Output Files

### Location
```
outputs/
└── <date>/
    └── <time>/
        ├── stratified_selection_analysis.png
        ├── random_vs_stratified_comparison.png
        ├── comparison_metrics.png
        ├── summary_comparison.png
        ├── results.pkl
        └── .hydra/
            └── config.yaml
```

### Plots

1. **`stratified_selection_analysis.png`**
   - Stratum participation over rounds
   - Actual vs expected representation
   - Deviation heatmap
   - Fairness metrics

2. **`random_vs_stratified_comparison.png`**
   - R², RMSE, MAE comparison
   - Performance improvements

3. **`comparison_metrics.png`**
   - Metric evolution over rounds

4. **`summary_comparison.png`**
   - Final metrics bar chart

## 📈 Expected Metrics

### Fairness Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| Gini Coefficient | < 0.15 | < 0.10 |
| Representation Ratio | 0.8 - 1.2 | 0.9 - 1.1 |
| Toxic Round Frequency | < 15% | < 10% |

### Performance Improvements (Stratified vs Random)

| Metric | Expected Improvement |
|--------|---------------------|
| R² Score | +0.5% to +1.0% |
| Min Stratum R² | +3% to +7% |
| Toxic Rounds | -60% to -80% |
| Gradient Variance | -25% to -40% |
| Fairness Gap | -40% to -60% |

## 🔧 Common Commands

### Basic Experiments

```bash
# Default run
python main.py strategy=compare_stratified

# High heterogeneity
python main.py strategy=compare_stratified alpha=0.1

# More rounds
python main.py strategy=compare_stratified num_rounds=50

# More clients per round
python main.py strategy=compare_stratified num_clients_per_round_fit=9
```

### Combined Parameters

```bash
# High heterogeneity + extended training
python main.py strategy=compare_stratified alpha=0.1 num_rounds=50

# More clients + stricter fairness
python main.py strategy=compare_stratified num_clients_per_round_fit=9 min_clients_per_stratum=2

# Full custom configuration
python main.py strategy=compare_stratified \
  alpha=0.2 \
  num_rounds=30 \
  num_clients_per_round_fit=9 \
  min_clients_per_stratum=2 \
  batch_size=64
```

### Other Strategies

```bash
# Standard FedAvg baseline
python main.py strategy=fedavg num_rounds=20

# FedProx with adaptive mu
python main.py strategy=compare_adaptive num_rounds=15
```

## 📁 File Structure

```
flowertry/
├── main.py                              # Main entry point
├── dataset.py                           # Data loading & partitioning
├── client.py                            # Flower client implementation
├── model.py                             # Neural network model
├── server.py                            # Server configuration
├── strategy.py                          # AdaptiveFedProx strategy
├── stratified_selector.py               # ✨ Stratified selection logic
├── stratified_strategy.py               # ✨ StratifiedFedProx strategy
├── plotting.py                          # Visualization utilities
├── adaptive_mu.py                       # Adaptive mu controller
├── conf/
│   └── base.yaml                        # Configuration file
├── data/
│   └── IndianPersonalFinance/
│       └── indianPersonalFinanceAndSpendingHabits.csv
├── outputs/                             # Generated results
├── STRATIFIED_SELECTION_README.md       # Complete documentation
├── IMPLEMENTATION_SUMMARY.md            # Implementation overview
├── ARCHITECTURE.md                      # System architecture
└── QUICK_REFERENCE.md                   # This file
```

## 🎯 Dataset Information

### Structure
- **Total Samples**: 19,483
- **Features**: 19 (12 numerical + 7 categorical)
- **Target**: Disposable_Income (regression)

### Partitioning
- **Total Clients**: 12
- **Strata**: 3 (City Tiers)
  - Tier 1: 4 clients (High income, urban)
  - Tier 2: 4 clients (Medium income, mixed)
  - Tier 3: 4 clients (Lower income, rural)

### Heterogeneity
- **Income CV**: 96.22% (very high)
- **Alpha Control**: 0.1 (high) to 1.0 (low)

## 🔍 Troubleshooting

### Error: "clients_per_round must be >= num_strata * min_per_stratum"

**Fix**: Increase clients per round
```bash
python main.py num_clients_per_round_fit=6  # At least 3 × 1 = 3
```

### High Toxic Round Frequency (>20%)

**Fix**: Increase clients per round for better balance
```bash
python main.py num_clients_per_round_fit=9
```

### Unbalanced Representation

**Fix**: Adjust alpha parameter
```bash
python main.py alpha=0.3  # More balanced mixing
```

### No Plots Generated

**Check**:
1. `conf/base.yaml`: `plotting.save_plots: true`
2. Output directory permissions
3. Matplotlib backend

## 📊 Interpreting Results

### Console Output

```
Round 1 - Stratified Selection:
  Selected 6 clients: [0, 2, 4, 6, 8, 10]
  Stratum distribution: {'Tier_1': 2, 'Tier_2': 2, 'Tier_3': 2}
```

**Good**: All strata represented (2 clients each)

```
STRATIFIED SELECTION FINAL SUMMARY
Participation equity (Gini): 0.0523
Representation ratios:
  Tier_1: 1.017
  Tier_2: 0.992
  Tier_3: 1.008
Toxic round frequency: 5.0%
```

**Excellent**: Low Gini, ratios near 1.0, low toxic frequency

### Plot Interpretation

**Stacked Area Chart**:
- Smooth, balanced layers = good stratification
- Spiky, uneven layers = poor balance

**Deviation Heatmap**:
- Green = over-represented
- Red = under-represented
- Yellow/white = balanced (ideal)

**Box Plot**:
- Similar boxes across strata = fair participation
- Wide boxes = high variance (not ideal)

## 💡 Best Practices

### For High Heterogeneity

```bash
python main.py strategy=compare_stratified \
  alpha=0.1 \
  num_clients_per_round_fit=9 \
  min_clients_per_stratum=2
```

### For Quick Testing

```bash
python main.py strategy=compare_stratified \
  num_rounds=10 \
  num_clients_per_round_fit=6
```

### For Production/Research

```bash
python main.py strategy=compare_stratified \
  alpha=0.3 \
  num_rounds=50 \
  num_clients_per_round_fit=9 \
  batch_size=64
```

## 🔗 Related Commands

### View Configuration

```bash
# Show current config
cat conf/base.yaml

# Show Hydra output config
cat outputs/<date>/<time>/.hydra/config.yaml
```

### Check Results

```bash
# List output directories
ls -lt outputs/

# View latest results
ls -lt outputs/$(ls -t outputs/ | head -1)/$(ls -t outputs/$(ls -t outputs/ | head -1) | head -1)/
```

### Python API Usage

```python
from stratified_selector import StratifiedClientSelector

# Create selector
selector = StratifiedClientSelector(
    client_strata={
        'Tier_1': [0, 1, 2, 3],
        'Tier_2': [4, 5, 6, 7],
        'Tier_3': [8, 9, 10, 11]
    },
    clients_per_round=6,
    min_per_stratum=1
)

# Select clients
selected = selector.select_clients(round_num=1)
print(f"Selected: {selected}")

# Get statistics
stats = selector.get_stratum_statistics(selected)
print(f"Stats: {stats}")

# Compute fairness metrics
metrics = selector.compute_fairness_metrics()
print(f"Gini: {metrics['participation_equity_gini']:.4f}")
```

## 📚 Documentation

- **Complete Guide**: `STRATIFIED_SELECTION_README.md`
- **Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `ARCHITECTURE.md`
- **Research**: `stratified_client_selection_strategy.md.pdf`

## 🎓 Research Context

### Theory
- Based on Cochran (1977) stratified sampling
- Maintains FedAvg convergence: O(1/√(KT))
- Variance reduction: 30-50% under high heterogeneity

### Contributions
- Bridges survey statistics and federated learning
- Fairness guarantees for minority groups
- Simple implementation (<100 lines core logic)

### Citations

```bibtex
@article{cochran1977sampling,
  title={Sampling techniques},
  author={Cochran, William G},
  year={1977}
}

@inproceedings{mcmahan2017communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and others},
  booktitle={AISTATS},
  year={2017}
}
```

---

**Quick Reference Version**: 1.0  
**Last Updated**: January 23, 2026  
**For**: Personal Finance FL with Stratified Selection
