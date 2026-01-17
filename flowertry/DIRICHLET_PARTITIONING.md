# Dirichlet Partitioning Guide

## Feature Implementation Complete ✅

You can now control data heterogeneity using Dirichlet distribution with the `alpha` parameter.

## How to Use

### Option 1: Edit Config File (base.yaml)

```yaml
# Set partition strategy to dirichlet
partition_strategy: dirichlet

# Control heterogeneity with alpha (0.1 to 10.0+)
dirichlet_alpha: 0.5 # Change this value!
```

Then run normally:

```bash
python main.py
```

### Option 2: Command Line Override

```bash
# High heterogeneity (extreme non-IID)
python main.py partition_strategy=dirichlet dirichlet_alpha=0.1

# Moderate heterogeneity
python main.py partition_strategy=dirichlet dirichlet_alpha=1.0

# Low heterogeneity (approaching IID)
python main.py partition_strategy=dirichlet dirichlet_alpha=10.0
```

## Alpha Parameter Guide

| Alpha Value | Heterogeneity Level  | Sample Imbalance    | Use Case                                 |
| ----------- | -------------------- | ------------------- | ---------------------------------------- |
| **0.1**     | **Extreme non-IID**  | Very high (50-100x) | Test robustness to extreme heterogeneity |
| **0.3-0.5** | **High non-IID**     | High (20-50x)       | Realistic challenging FL scenario        |
| **1.0**     | **Moderate non-IID** | Moderate (5-20x)    | Typical FL heterogeneity                 |
| **3.0-5.0** | **Low non-IID**      | Low (2-5x)          | Mild heterogeneity                       |
| **10.0+**   | **Near IID**         | Very low (<2x)      | Almost balanced data                     |

## What Dirichlet Controls

1. **Sample Distribution**: How many samples each client gets
   - Low alpha → Very imbalanced (one client may get 50%+ of data)
   - High alpha → More balanced distribution

2. **Data Skewness**: Natural variation in target distribution
   - Lower alpha creates more diverse client populations
   - Higher alpha creates more similar clients

## Example Runs

### Compare Heterogeneity Levels

```bash
# Extreme non-IID
python main.py partition_strategy=dirichlet dirichlet_alpha=0.1 num_rounds=25

# High non-IID (recommended for testing)
python main.py partition_strategy=dirichlet dirichlet_alpha=0.5 num_rounds=25

# Moderate non-IID
python main.py partition_strategy=dirichlet dirichlet_alpha=1.0 num_rounds=25

# Near IID
python main.py partition_strategy=dirichlet dirichlet_alpha=10.0 num_rounds=25
```

### Compare with Existing Strategies

```bash
# Hybrid partitioning (City_Tier × Occupation)
python main.py partition_strategy=hybrid

# Dirichlet with similar heterogeneity
python main.py partition_strategy=dirichlet dirichlet_alpha=0.5
```

## Important Notes

1. **Minimum Samples**: The implementation ensures each client gets at least 50 samples to avoid empty clients

2. **Seed Control**: Use the same seed for reproducible partitioning:

   ```yaml
   seed: 2023 # In config file
   ```

3. **Number of Clients**: Works with any number of clients:

   ```yaml
   num_clients: 10 # Can be 3, 5, 10, 20, etc.
   ```

4. **Strategy Comparison**: Can still compare with other strategies:
   ```yaml
   compare_all: true
   strategies:
     - fedavg
     - fedprox
     - hybrid
   ```

## Verification

To test different alpha values:

```bash
python test_dirichlet.py
```

This will show heterogeneity metrics for alphas: 0.1, 0.5, 1.0, 5.0, 10.0

## Output Information

When running with Dirichlet partitioning, you'll see:

- Sample distribution per client
- Sample imbalance ratio (max/min)
- Target distribution statistics
- Heterogeneity coefficient of variation

## Research Use

This implementation follows the standard Dirichlet partitioning method used in FL research papers:

- Lower alpha = more challenging non-IID scenario
- Can directly compare results with published FL benchmarks
- Allows systematic evaluation of FL algorithms under varying heterogeneity
