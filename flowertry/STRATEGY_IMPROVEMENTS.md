# Federated Learning Strategy Improvements

## Overview

This document summarizes the improvements made to FedProx, SCAFFOLD, and Hybrid strategies to match the performance gains achieved with FedAvg.

## Key Improvements

### 1. **Modern Optimizer Integration**

- **Before**: Manual SGD parameter updates
- **After**: AdamW optimizer with adaptive learning rates
  - Decoupled weight decay (1e-4)
  - Adaptive per-parameter learning rates
  - Better handling of sparse gradients in non-IID data

### 2. **Learning Rate Scheduling**

- **Cosine Annealing**: Smooth learning rate decay from `lr` to `lr * 0.1`
  - `T_max = len(trainloader) * local_epochs`
  - Prevents sudden drops in learning capacity
  - Better convergence in final rounds

### 3. **Gradient Clipping**

- **Max norm**: 1.0
- **Purpose**: Prevent exploding gradients in heterogeneous data
- Applied after gradient corrections (SCAFFOLD) and regularization (FedProx)

### 4. **Optimized Hyperparameters**

#### FedProx

- **mu**: 0.05 (reduced from 0.1)
- **Rationale**: AdamW + improved model need less aggressive regularization
- Proximal term still provides stability without hindering convergence

#### SCAFFOLD

- **Control Variate Aggregation**: Properly implemented
  - Server control update: `c = c + (1/N) * Σ delta_c_i`
  - Client control update: `c_i_new = c_i - c + (x_0 - x_K) / (K * lr)`
- **Gradient Correction**: Applied before optimizer step
  - `g_corrected = g_local + (c - c_i)`
  - Reduces variance in heterogeneous data

#### Hybrid (FedProx + SCAFFOLD)

- **fedprox_weight**: 0.3 (moderate influence)
- **scaffold_weight**: 0.5 (stronger variance reduction)
- **mu**: 0.03 (even lower with combined regularization)
- **Warmup**: 5 rounds with cosine schedule
  - Gradual activation of regularization
  - Prevents over-regularization in early rounds

## Technical Details

### SCAFFOLD Control Variate Flow

```
Client Training:
1. Receive server_control (c) and use client_control (c_i)
2. For each gradient: g_corrected = g_local + (c - c_i)
3. Apply gradient clipping: clip_grad_norm(params, 1.0)
4. Update with AdamW: optimizer.step()
5. Update client control: c_i_new = c_i - c + (x_0 - x_K) / (K * lr)
6. Return delta_control = c_i_new - c_i

Server Aggregation:
1. Aggregate parameters: w_new = weighted_average(w_clients)
2. Aggregate control variates: c_new = c_old + (1/N) * Σ delta_c_i
3. Broadcast w_new and c_new to clients
```

### Hybrid Strategy Composition

```
Loss = MSE_loss + fedprox_weight * (mu/2) * ||w - w_global||^2

Gradient Correction:
g_corrected = g_loss + fedprox_weight * mu * (w - w_global)  # FedProx proximal
            + scaffold_weight * (c - c_i)                    # SCAFFOLD correction

This allows fine-grained control:
- fedprox_weight=0, scaffold_weight=1 → Pure SCAFFOLD
- fedprox_weight=1, scaffold_weight=0 → Pure FedProx
- Both > 0 → Hybrid with complementary effects
```

## Performance Expectations

### FedAvg (Baseline)

- **Strengths**: Simple, fast convergence with IID data
- **Weaknesses**: Slow convergence with heterogeneous data

### FedProx (Improved)

- **Strengths**: Handles heterogeneity via proximal term
- **Expected MAPE**: 5-10% better than FedAvg on non-IID data
- **Convergence**: Smoother, more stable

### SCAFFOLD (Improved)

- **Strengths**: Corrects client drift via control variates
- **Expected MAPE**: 10-15% better than FedAvg on highly non-IID data
- **Convergence**: Faster in later rounds as control variates converge

### Hybrid (Improved)

- **Strengths**: Combines stability (FedProx) with drift correction (SCAFFOLD)
- **Expected MAPE**: 15-20% better than FedAvg on challenging scenarios
- **Convergence**: Most stable across all data distributions

## Configuration Usage

### Test All Strategies

```bash
python main.py --config-name=improved_strategies
```

### Test Individual Strategy

```bash
python main.py strategy=fedprox fedprox.mu=0.05
python main.py strategy=scaffold
python main.py strategy=hybrid hybrid.fedprox_weight=0.3 hybrid.scaffold_weight=0.5
```

### Hyperparameter Tuning

```bash
# Test different mu values for FedProx
python main.py strategy=fedprox fedprox.mu=0.01
python main.py strategy=fedprox fedprox.mu=0.05
python main.py strategy=fedprox fedprox.mu=0.1

# Test different weight combinations for Hybrid
python main.py strategy=hybrid hybrid.fedprox_weight=0.5 hybrid.scaffold_weight=0.5
python main.py strategy=hybrid hybrid.fedprox_weight=0.3 hybrid.scaffold_weight=0.7
python main.py strategy=hybrid hybrid.fedprox_weight=0.7 hybrid.scaffold_weight=0.3
```

## Monitoring

### Key Metrics to Track

1. **Training Loss**: Should decrease smoothly
2. **Validation MAPE**: Primary performance metric
3. **R² Score**: Model explanation power
4. **MAE**: Absolute error magnitude
5. **SCAFFOLD Metrics** (SCAFFOLD/Hybrid only):
   - `scaffold_updates`: Number of successful control updates
   - `avg_control_delta_norm`: Magnitude of control changes

### Expected Behavior

**Early Rounds (1-10)**:

- FedAvg: Fast initial improvement
- FedProx: Slightly slower but more stable
- SCAFFOLD: Slower as control variates initialize
- Hybrid: Moderate, balanced convergence

**Mid Rounds (10-20)**:

- FedAvg: May plateau with non-IID data
- FedProx: Steady improvement
- SCAFFOLD: Accelerating as controls converge
- Hybrid: Consistent gains

**Late Rounds (20-30)**:

- FedAvg: Minimal improvement
- FedProx: Continued gradual improvement
- SCAFFOLD: Best performance on heterogeneous data
- Hybrid: Overall best results

## Troubleshooting

### SCAFFOLD Metrics Missing

**Symptom**: `scaffold_updates` not in aggregated metrics
**Solution**: Ensure `delta_control` is properly serialized in client.py fit() method

### Hybrid Underperforming

**Symptom**: Hybrid worse than individual strategies
**Possible Causes**:

1. Weight imbalance (adjust fedprox_weight and scaffold_weight)
2. Too aggressive warmup (increase warmup_rounds)
3. mu too high (reduce to 0.01-0.03 range)

### Control Variates Not Converging

**Symptom**: `avg_control_delta_norm` increasing over rounds
**Possible Causes**:

1. Learning rate too high (reduce from 0.001)
2. Data too heterogeneous (increase num_rounds)
3. Batch size too small (check data splits)

## Next Steps

1. **Run Comparison**: Execute all strategies and compare MAPE
2. **Hyperparameter Tuning**: Fine-tune based on initial results
3. **Data Distribution Analysis**: Check if non-IID assumptions hold
4. **Per-Client Metrics**: Identify which clients benefit most from each strategy

## References

- **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
- **SCAFFOLD**: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for FL" (ICML 2020)
- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019)
