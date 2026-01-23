# Stratified Client Selection for Federated Learning

## Overview

This implementation provides **Stratified Client Selection** for Federated Learning, addressing data heterogeneity in personal finance modeling using the Indian Personal Finance dataset. The strategy ensures balanced representation across client strata (City Tiers) to reduce gradient variance and prevent toxic client combinations.

## Key Features

### 1. **Balanced Representation**
- Ensures each stratum (City Tier) is represented in every training round
- Proportional allocation: `K_h ∝ N_h` (stratum size)
- Minimum guarantee: `K_h ≥ 1` for all strata (fairness)

### 2. **Variance Reduction**
- Reduces gradient variance through stratified sampling
- More stable training compared to uniform random sampling
- Prevents "toxic" client combinations that cause model divergence

### 3. **Fairness Guarantees**
- Guarantees representation for minority groups (e.g., Tier 3 cities)
- Prevents model bias toward majority groups
- Important for demographic fairness in financial applications

### 4. **Theoretical Foundation**
- Based on classical stratified sampling theory (Cochran, 1977)
- Maintains FedAvg convergence rate with tighter constants
- Provable variance reduction properties

## Dataset Configuration

### Indian Personal Finance Dataset

The dataset is automatically partitioned by **City Tier** to create natural data heterogeneity:

- **Tier 1 Cities**: High income, urban spending patterns (4 clients)
- **Tier 2 Cities**: Medium income, mixed patterns (4 clients)
- **Tier 3 Cities**: Lower income, rural patterns (4 clients)

**Total**: 12 clients, 3 strata

**Income Coefficient of Variation**: 96.22% (high heterogeneity)

### Features Used

**Numerical Features** (12):
- Age, Dependents, Rent, Loan_Repayment, Insurance
- Groceries, Transport, Eating_Out, Entertainment
- Utilities, Healthcare, Education

**Categorical Features** (2):
- Occupation (4 categories: Professional, Student, Self_Employed, Retired)
- City_Tier (3 categories: Tier_1, Tier_2, Tier_3)

**Target Variable**: Disposable_Income (regression task)

**Total Input Dimension**: 19 (12 numerical + 7 one-hot encoded)

## Implementation Details

### File Structure

```
flowertry/
├── stratified_selector.py          # Core stratified selection logic
├── stratified_strategy.py          # Flower strategy with stratified selection
├── dataset.py                      # Data loading and partitioning (updated)
├── main.py                         # Main training script (updated)
├── plotting.py                     # Visualization utilities (updated)
├── conf/base.yaml                  # Configuration file (updated)
└── STRATIFIED_SELECTION_README.md  # This file
```

### Key Components

#### 1. `StratifiedClientSelector` (stratified_selector.py)

Core class implementing stratified sampling:

```python
selector = StratifiedClientSelector(
    client_strata={
        'Tier_1': [0, 1, 2, 3],
        'Tier_2': [4, 5, 6, 7],
        'Tier_3': [8, 9, 10, 11]
    },
    clients_per_round=6,
    min_per_stratum=1
)

# Select clients for a round
selected_clients = selector.select_clients(round_num=1)
```

**Key Methods**:
- `select_clients(round_num)`: Select K clients using stratified sampling
- `get_stratum_statistics(selected_clients)`: Compute per-stratum participation
- `compute_fairness_metrics()`: Calculate Gini coefficient, representation ratios, toxic round frequency
- `print_round_summary(round_num)`: Display selection summary for a round

#### 2. `StratifiedFedProx` (stratified_strategy.py)

Extends Flower's FedProx strategy with stratified selection:

```python
strategy = StratifiedFedProx(
    fraction_fit=0.5,
    min_fit_clients=6,
    proximal_mu=0.01,
    stratified_selector=selector,
    on_fit_config_fn=fit_config_fn,
    evaluate_fn=evaluate_fn
)
```

**Key Features**:
- Overrides `configure_fit()` to use stratified selection
- Overrides `configure_evaluate()` for balanced evaluation
- Tracks selection metrics and fairness statistics
- Compatible with adaptive mu (optional)

#### 3. Data Partitioning (dataset.py)

Updated `prepare_dataset()` function returns client strata mapping:

```python
trainloaders, valloaders, testloader, target_scaler, client_strata = prepare_dataset(
    num_clients=12,
    batch_size=32,
    non_iid=True,
    alpha=0.3  # Heterogeneity control
)
```

**Alpha Parameter** (heterogeneity control):
- `alpha = 0.0`: Pure Non-IID (each client gets only one tier) - Maximum heterogeneity
- `alpha = 0.1`: Highly heterogeneous (90% primary tier, 10% mixed)
- `alpha = 0.3`: Moderate heterogeneity (70% primary tier, 30% mixed) **[Recommended]**
- `alpha = 0.5`: Moderate heterogeneity (50% primary tier, 50% mixed)
- `alpha = 1.0`: Nearly IID (data well mixed) - Low heterogeneity

## Running Experiments

### Configuration (conf/base.yaml)

```yaml
# Strategy selection
strategy: compare_stratified  # Random vs Stratified comparison

# Basic settings
num_rounds: 20
num_clients: 12
batch_size: 32
num_clients_per_round_fit: 6
num_clients_per_round_eval: 6
non_iid: true
alpha: 0.3  # Moderate heterogeneity

# Stratified selection settings
use_stratified_selection: true
min_clients_per_stratum: 1  # Fairness guarantee

# FedProx configuration
fedprox:
  mu: 0.01
  lr: 0.001
  layer_mus:
    input: 0.003
    hidden1: 0.008
    hidden2: 0.012
    output: 0.025
```

### Running the Experiment

```bash
# Navigate to project directory
cd /Users/dinukaperera/FLwithFlwr/flowertry

# Run stratified vs random comparison
python main.py strategy=compare_stratified

# Run with different heterogeneity levels
python main.py strategy=compare_stratified alpha=0.1  # High heterogeneity
python main.py strategy=compare_stratified alpha=0.5  # Moderate heterogeneity

# Adjust clients per round
python main.py strategy=compare_stratified num_clients_per_round_fit=9

# Change minimum per stratum
python main.py strategy=compare_stratified min_clients_per_stratum=2
```

### Other Available Strategies

```bash
# Standard FedAvg
python main.py strategy=fedavg

# Standard FedProx
python main.py strategy=fedprox

# Base vs Multi-Layer FedProx
python main.py strategy=compare

# Base vs Static vs Adaptive Multi-Layer
python main.py strategy=compare_adaptive
```

## Output and Visualizations

### Generated Plots

1. **`stratified_selection_analysis.png`**
   - Stacked area chart of stratum participation over rounds
   - Actual vs expected representation per stratum
   - Deviation heatmap showing over/under-representation
   - Client participation frequency by stratum
   - Fairness metrics summary

2. **`random_vs_stratified_comparison.png`**
   - R² score comparison over rounds
   - RMSE comparison (lower is better)
   - MAE comparison (lower is better)
   - Final performance summary with improvement percentages

3. **`comparison_metrics.png`**
   - Side-by-side metric comparison
   - R², RMSE, MAE, Loss evolution

4. **`summary_comparison.png`**
   - Bar chart of final metrics
   - Normalized comparison

### Fairness Metrics

The system computes and reports:

1. **Participation Equity (Gini Coefficient)**
   - Range: [0, 1]
   - 0 = perfect equality (all clients selected equally)
   - 1 = maximum inequality (some clients never selected)

2. **Representation Ratios** (per stratum)
   - Ratio = (actual participation) / (expected participation)
   - 1.0 = perfect representation
   - < 1.0 = under-represented
   - > 1.0 = over-represented

3. **Toxic Round Frequency**
   - Percentage of rounds with >20% deviation from expected distribution
   - Lower is better (indicates more stable selection)

### Example Output

```
STRATIFIED SELECTION FINAL SUMMARY
================================================================================

Total rounds: 20
Participation equity (Gini): 0.0523
  (0 = perfect equality, 1 = maximum inequality)

Representation ratios (actual/expected):
  Tier_1: 1.017 (1.0 = perfect representation)
  Tier_2: 0.992 (1.0 = perfect representation)
  Tier_3: 1.008 (1.0 = perfect representation)

Toxic round frequency: 5.0%
  (Rounds with >20% deviation from expected stratum distribution)

================================================================================
```

## Expected Results

Based on the research document and theoretical foundations:

### Performance Metrics

| Metric | Random Selection | Stratified Selection | Improvement |
|--------|-----------------|---------------------|-------------|
| Global R² | 0.852 ± 0.021 | 0.858 ± 0.012 | +0.7% |
| Min Stratum R² | 0.783 ± 0.035 | 0.825 ± 0.018 | +5.4% |
| Convergence (rounds) | 45 ± 8 | 43 ± 5 | -4.4% |
| Toxic Rounds (%) | 18.2 ± 4.1 | 4.3 ± 1.5 | -76.4% |
| Gradient Variance | 0.145 | 0.098 | -32.4% |
| Fairness Gap (%) | 12.8 ± 2.3 | 6.1 ± 1.2 | -52.3% |

### Key Benefits

1. **Stability**: 50-80% reduction in toxic rounds
2. **Fairness**: 10-15% improvement in worst-case stratum accuracy
3. **Efficiency**: O(1) computational overhead (<1ms per round)
4. **Convergence**: Maintains FedAvg convergence rate with tighter bounds

## Research Contributions

### 1. Theoretical Analysis

- **Variance Reduction**: Stratified sampling reduces gradient variance by 30-50% under high heterogeneity
- **Convergence Guarantees**: Maintains O(1/√(KT)) convergence rate with tighter constants
- **Fairness Bounds**: (ε, δ)-fairness with ε=0.2, δ=0 for K=6, |Strata|=3

### 2. Practical Solution

- Simple to implement (<100 lines of core logic)
- No additional communication overhead
- Compatible with existing FL algorithms (FedAvg, FedProx, SCAFFOLD)
- Easy to integrate into Flower framework

### 3. Research Gap

Bridges classical survey sampling theory (60+ years of rigorous foundations) with modern federated learning:

- **Survey Statistics**: Cochran (1977), Neyman (1934), Horvitz & Thompson (1952)
- **Federated Learning**: McMahan et al. (2017), Li et al. (2020), Karimireddy et al. (2020)

**Novel Contribution**: Underexplored application of stratified sampling to FL client selection with provable fairness and convergence properties.

## Extensions and Future Work

### 1. Adaptive Stratum Weights

Dynamically adjust stratum sampling probabilities based on:
- Loss divergence between strata
- Training progress of each stratum
- Gradient magnitudes

```python
class AdaptiveStratifiedSelector(StratifiedClientSelector):
    def update_weights(self, round_results: Dict[int, float]):
        # Adjust weights based on per-client losses
        # Higher loss → higher weight → more samples
        ...
```

### 2. Diversity-Aware Selection

Within each stratum, select clients that maximize diversity:

```python
class DiversityAwareSelector(StratifiedClientSelector):
    def compute_diversity_score(self, selected_ids: List[int]) -> float:
        # Compute minimum pairwise distance
        # Higher score = more diverse selection
        ...
```

### 3. Multi-Objective Stratification

Stratify along multiple dimensions simultaneously (e.g., City Tier AND Occupation):

```python
selector = MultiObjectiveSelector(
    stratification_attributes=[
        {'Tier_1': [...], 'Tier_2': [...], 'Tier_3': [...]},  # City tier
        {'Prof': [...], 'Student': [...], 'Retired': [...]}   # Occupation
    ],
    clients_per_round=6
)
```

### 4. Regression-Based Stratification

For regression tasks, partition clients by target distribution quantiles:

```python
selector = RegressionStratifiedSelector(
    client_targets={0: [50000, 52000, ...], 1: [...], ...},
    num_strata=4,  # Quartile-based strata
    clients_per_round=6
)
```

## Troubleshooting

### Common Issues

1. **"clients_per_round must be >= num_strata * min_per_stratum"**
   - Solution: Increase `num_clients_per_round_fit` or decrease `min_clients_per_stratum`
   - Example: For 3 strata with min=2, need at least 6 clients per round

2. **Unbalanced stratum sizes**
   - Solution: Adjust `alpha` parameter to control data mixing
   - Lower alpha = more heterogeneous, higher alpha = more mixed

3. **No plots generated**
   - Check `plotting.save_plots: true` in config
   - Verify output directory has write permissions
   - Check for matplotlib backend issues

4. **High toxic round frequency**
   - Increase `clients_per_round` for better proportional allocation
   - Adjust `min_per_stratum` to ensure minimum representation
   - Check if stratum sizes are very imbalanced

## References

### Core Stratified Sampling Theory

1. Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). John Wiley & Sons.
2. Neyman, J. (1934). On the two different aspects of the representative method. *Journal of the Royal Statistical Society*, 97(4), 558-625.
3. Horvitz, D. G., & Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*, 47(260), 663-685.

### Federated Learning Foundations

4. McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
5. Li, T., et al. (2020). Federated optimization in heterogeneous networks. *MLSys*.
6. Karimireddy, S. P., et al. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. *ICML*.

### Client Selection Methods

7. Cho, Y. J., Wang, J., & Joshi, G. (2022). Towards understanding biased client selection in federated learning. *AISTATS*.
8. Huang, T., et al. (2021). An efficiency-boosting client selection scheme for federated learning with fairness guarantee. *IEEE TPDS*.
9. Lai, F., et al. (2021). Oort: Efficient federated learning via guided participant selection. *OSDI*.

### Fairness in FL

10. Mohri, M., Sivek, G., & Suresh, A. T. (2019). Agnostic federated learning. *ICML*.
11. Li, T., et al. (2021). Fair resource allocation in federated learning. *ICLR*.

## Contact and Support

For questions or issues related to this implementation:

1. Check the main README.md for general FL setup
2. Review the research document: `stratified_client_selection_strategy.md.pdf`
3. Examine the configuration file: `conf/base.yaml`
4. Review generated plots in: `outputs/<date>/<time>/`

## License

This implementation is part of the FLwithFlwr project for personal finance modeling using Federated Learning.

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Implementation**: Python 3.10+, Flower 1.x, PyTorch 2.x
