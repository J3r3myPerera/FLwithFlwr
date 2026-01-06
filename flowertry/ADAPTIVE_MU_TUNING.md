# Adaptive FedProx Mu Tuning Guide

## Why FedAvg Performed Better Than Adaptive FedProx

Based on your results showing FedAvg (59.45%) outperforming FedProx Adaptive (47.65%), here are the key issues identified:

### Problem 1: Over-Regularization (Mu Reached Maximum)
- **Issue**: Mu reached 1.0 (the maximum), which is **extremely high** regularization
- **Impact**: A mu of 1.0 prevents the model from learning effectively by forcing client updates to stay too close to the global model
- **Typical Range**: FedProx typically uses mu in the range **0.01-0.5**, with most successful experiments using **0.1-0.3**

### Problem 2: Too Sensitive to Loss Fluctuations
- **Issue**: `loss_threshold: 0.0` means ANY increase in loss triggers mu increase
- **Impact**: Normal training fluctuations (which are expected) cause unnecessary mu increases
- **Solution**: Use a threshold (e.g., 0.01) to ignore small noise

### Problem 3: Aggressive Increase Factor
- **Issue**: `increase_factor: 1.5` causes rapid mu growth (50% increase per adaptation)
- **Impact**: Mu escalates too quickly, reaching maximum too early
- **Solution**: Use a smaller factor (e.g., 1.2) for more gradual adaptation

### Problem 4: Asymmetric Adaptation
- **Issue**: Mu increases by 50% (×1.5) but decreases by only 10% (×0.9)
- **Impact**: Once mu increases, it's hard to bring it back down
- **Solution**: Balance the factors (e.g., 1.2 increase, 0.95 decrease)

## Updated Configuration

The configuration has been updated with more conservative parameters:

```yaml
adaptive_mu:
  enabled: true
  initial_mu: 0.1         # Good starting point
  mu_min: 0.01            # Increased from 0.001 for stability
  mu_max: 0.5             # Reduced from 1.0 (typical FedProx range)
  increase_factor: 1.2    # Reduced from 1.5 (less aggressive)
  decrease_factor: 0.95   # Increased from 0.9 (faster recovery)
  loss_threshold: 0.01    # Increased from 0.0 (ignore noise)
  warmup_rounds: 5        # Increased from 3 (more stable start)
```

## Improved Adaptive Mechanism

The adaptation logic has been enhanced to:
1. **Use relative loss changes**: Considers percentage change, not just absolute
2. **Scale adaptation by magnitude**: Larger loss increases trigger proportionally larger mu increases
3. **Ignore small fluctuations**: Only adapts when loss change is significant (>5% relative or >threshold absolute)

## When FedProx Should Outperform FedAvg

FedProx is particularly beneficial when:
1. **High data heterogeneity** (non-IID data with low alpha, e.g., 0.1-0.3)
2. **Many local epochs** (more local training = more divergence risk)
3. **Large number of clients** (more heterogeneity in updates)
4. **Convergence issues** (FedAvg struggling to converge)

## Recommendations

1. **Start with fixed mu**: Try `proximal_mu: 0.1` or `0.25` first to establish baseline
2. **Use adaptive mu conservatively**: Start with the updated parameters above
3. **Monitor mu evolution**: Check that mu stays in reasonable range (0.01-0.5)
4. **Compare with fixed mu**: Run both fixed and adaptive to see which works better for your data

## Expected Behavior After Fixes

With the updated configuration:
- Mu should stay in the range 0.01-0.5
- Adaptation should be more gradual and stable
- Mu should decrease when model converges (loss decreases)
- Better balance between regularization and learning

Run the experiment again with these updated parameters to see improved performance!

