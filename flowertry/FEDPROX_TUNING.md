# FedProx Performance Tuning Guide

## Why FedAvg Might Outperform FedProx

While FedProx is designed to handle non-IID data better, several factors can cause FedAvg to perform better:

### 1. **Proximal Mu (μ) Parameter**
The `proximal_mu` parameter controls the strength of the regularization term:
- **Too small (e.g., 0.01)**: Weak regularization, behaves like FedAvg
- **Too large (e.g., 1.0)**: Strong regularization, may prevent convergence
- **Optimal range**: Typically 0.01-0.5, with 0.1-0.3 being common

**Current setting**: `proximal_mu: 0.1` - This is reasonable but may need tuning.

### 2. **Number of Rounds**
FedProx benefits often appear after more rounds:
- **10 rounds**: May not be enough to see benefits
- **20-50 rounds**: Better to observe convergence differences
- **Recommendation**: Try 20-30 rounds for fair comparison

### 3. **Data Heterogeneity Level**
With `alpha=0.1` (very heterogeneous):
- Each client has mostly one class
- FedProx should help, but benefits may be subtle
- Try `alpha=0.5` for moderate heterogeneity to see clearer differences

### 4. **Learning Rate**
FedProx may need different learning rates:
- Current: `lr: 0.01`
- Try: `lr: 0.005` or `lr: 0.02` for FedProx specifically

### 5. **Local Epochs**
More local epochs can amplify client drift:
- Current: `local_epochs: 3`
- With more epochs (5-10), FedProx benefits become more apparent

## Recommendations

### Option 1: Increase Rounds and Tune Mu
```yaml
num_rounds: 30
proximal_mu: 0.3  # Try higher values
```

### Option 2: Moderate Heterogeneity
```yaml
alpha: 0.5  # Less extreme heterogeneity
proximal_mu: 0.1
```

### Option 3: More Local Epochs (to see drift)
```yaml
config_fit:
  local_epochs: 5  # More epochs = more drift = FedProx helps more
```

### Option 4: Lower Learning Rate for FedProx
FedProx often works better with slightly lower learning rates.

## Expected Behavior

- **FedAvg**: Fast initial convergence, may plateau or diverge with high heterogeneity
- **FedProx**: Slower initial convergence, but more stable and better final accuracy
- **FedSCAFFOLD**: Best for extreme heterogeneity, but more complex

## When FedAvg Can Outperform FedProx

1. **Mild heterogeneity**: If data is only slightly non-IID
2. **Small number of rounds**: FedAvg converges faster initially
3. **Suboptimal μ**: Wrong proximal_mu can hurt performance
4. **Simple models**: FedProx benefits are more apparent with complex models
5. **Small dataset**: With limited data, regularization can hurt

## Next Steps

1. Run with more rounds (20-30) to see convergence differences
2. Try different `proximal_mu` values: [0.01, 0.1, 0.3, 0.5]
3. Compare with moderate heterogeneity (`alpha=0.5`)
4. Increase local epochs to 5-10 to amplify client drift

