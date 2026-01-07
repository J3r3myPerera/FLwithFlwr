# FedProx Accuracy Improvement Factors

This document outlines various factors and strategies to improve FedProx performance over FedAvg, particularly in non-IID federated learning settings.

## 1. Hyperparameter Tuning

### Proximal Term Coefficient (μ)

**Key Points:**
- **Fixed μ**: Try values between **0.1-0.5** for non-IID settings
  - Too low (< 0.01): Minimal regularization, behaves like FedAvg
  - Too high (> 1.0): Over-regularization, slows convergence
- **Adaptive μ**: More aggressive adaptation
  - Increase `mu_max` to **2.0-5.0** for highly heterogeneous settings
  - Lower `smoothing_factor` (e.g., **0.3-0.5**) for faster adaptation
  - Reduce `warmup_rounds` to **1-2** to start regularization earlier

**Configuration Changes:**
```yaml
multi_signal_mu:
  base_mu: 0.01          # Conservative start
  mu_min: 0.001          # Minimum bound
  mu_max: 2.0            # Allow aggressive regularization
  smoothing_factor: 0.4  # Faster adaptation
  warmup_rounds: 2       # Start earlier
```

### Learning Rate and Optimization

**Recommendations:**
- **Lower Learning Rate**: Try **0.001-0.005** (vs default 0.01)
  - FedProx can handle slower convergence for better stability
- **Learning Rate Scheduling**: 
  - Exponential decay every 10-20 rounds
  - Step decay: reduce LR by 0.5-0.8 at milestones
  - Cosine annealing for smooth decay
- **Alternative Optimizers**:
  - **Adam/AdamW**: May work better with proximal term
  - **SGD with momentum**: Tune momentum (0.9-0.99)

**Example Configuration:**
```yaml
config_fit:
  lr: 0.005              # Reduced from 0.01
  momentum: 0.95         # Higher momentum
  lr_decay: 0.95         # Decay factor
  lr_decay_rounds: 10    # Decay every N rounds
```

## 2. Non-IID Data Settings

### Increase Heterogeneity

**Strategies:**
- **Lower α Parameter**: Use **α = 0.05-0.1** (very heterogeneous)
  - Current: α = 0.3 (moderate)
  - More extreme: α = 0.01 (most heterogeneous)
- **Extreme Non-IID**: 
  - Only 1-2 classes per client (pathological non-IID)
  - Test quantity skew (different dataset sizes)
  - Feature distribution shift
- **Client Participation Patterns**:
  - Some clients participate rarely
  - Simulate client dropouts mid-training

**Configuration:**
```yaml
iid: false
alpha: 0.05  # Very heterogeneous (was 0.3)
```

### Heterogeneity-Aware Client Selection

- Sample clients with higher heterogeneity metrics
- Weight client contributions by their heterogeneity
- Prioritize clients that diverge more from global model

## 3. Training Dynamics

### Local Epochs

**Key Insight:** FedProx shines when clients train longer locally (more drift potential)

- **Increase Local Epochs**: Try **30-50** (vs current 20)
  - More epochs = more client drift = more benefit from proximal term
- **Progressive Training**:
  - Start with fewer epochs (5-10), increase gradually
  - Or use adaptive epochs based on client data size
- **Per-Client Epochs**:
  - Larger clients: more epochs
  - Smaller clients: fewer epochs

**Configuration:**
```yaml
config_fit:
  local_epochs: 30  # Increased from 20
```

### Client Participation

- **More Clients Per Round**: 
  - Try 20-30 (vs current 10)
  - Better aggregation stability
- **Client Sampling Strategies**:
  - Ensure fair sampling across all client types
  - Stratified sampling by data distribution
- **Participation Rate**:
  - Test different participation fractions (0.3-0.8)
  - Higher participation = more stable but slower

## 4. Advanced Proximal Term Strategies

### Per-Layer μ

**Concept:** Different μ values for different layers

- **Lower μ** for early layers (feature extraction)
- **Higher μ** for later layers (task-specific)
- Focus regularization on layers with highest variance

**Implementation Idea:**
```python
layer_mu = {
    'fc1': 0.05,   # Feature extraction
    'fc2': 0.1,
    'fc3': 0.15,
    'fc4': 0.2     # Classification head
}
```

### Adaptive Per-Client μ

- Assign μ based on each client's heterogeneity
- Clients with skewed distributions get higher μ
- Clients with uniform distributions get lower μ

### Temporal μ Scheduling

- **Start Low, Increase Over Time**: 
  - Initial μ = 0.01, gradually increase to 0.1-0.5
- **Cyclic Schedule**: 
  - Increase μ when loss plateaus
  - Decrease when loss decreases significantly

## 5. Multi-Signal Adaptive Improvements

### Tune Signal Weights

**Current Weights:**
```yaml
weights:
  gradient_divergence: 0.35
  loss_variance: 0.25
  label_entropy: 0.25
  feature_variance: 0.15
```

**Recommended Adjustments:**
- **Increase gradient_divergence** to **0.4-0.5** (most direct measure)
- **Add new signals**:
  - Client participation rate
  - Local loss variance across batches
  - Update magnitude per client

### Better Signal Normalization

- Normalize signals by dataset size
- Use percentiles instead of means (more robust)
- Track signal trends over multiple rounds, not just current values

## 6. Aggregation and Model Updates

### Weighted Aggregation

- Weight by dataset size (already done in FedAvg)
- Weight by data quality or client reliability
- Use momentum in aggregation (FedAvgM)

### Regularization Techniques

- Combine FedProx with:
  - **Dropout**: May need adjustment with proximal term
  - **Batch Normalization**: Tune BN stats aggregation
- **L2 Regularization** on global model
- **Early Stopping**: Prevent over-regularization

## 7. Architecture and Model Factors

### Model Capacity

- **Deeper/Wider Networks**: More parameters = more drift potential
- **Model Pruning**: Sparser models may drift less

### Feature Normalization

- **Normalize Input Features**: Reduce distribution shifts
- **Batch Normalization**: 
  - FL-aware BN (aggregate BN stats)
  - Or use Layer Normalization (statistics-independent)

## 8. Experimental Setup

### Number of Rounds

- **More Rounds**: Try **100-200** rounds (vs current 50)
  - FedProx may be slower but more stable long-term
  - Convergence differences become more apparent

### Evaluation Strategy

- **Per-Client Metrics**: 
  - Track accuracy per client, not just global
  - Measure variance across clients
- **Convergence Speed**: 
  - Compare rounds to reach target accuracy
- **Stability**: 
  - Measure loss/accuracy variance over rounds
  - FedProx should show lower variance

### Baseline Comparison

- Ensure FedAvg is also optimally tuned
- Compare against FedAvg with same hyperparameters
- Account for computational cost differences

## 9. Dataset-Specific Considerations

### Class Imbalance

- FedProx helps more with **severe class imbalance**
- Test with imbalanced data distributions
- Measure per-class accuracy

### Feature Heterogeneity

- Ensure features have **varying distributions** across clients
- More heterogeneous features = more benefit from FedProx
- Consider domain adaptation scenarios

### Dataset Size

- **Larger local datasets**: More local training = more drift
- **Small clients**: May need different treatment
- Balance between client data sizes

## 10. Implementation Details

### Gradient Clipping Interaction

- **Lower Clipping**: Try `max_grad_norm = 0.5` (vs 1.0)
  - Works differently with proximal term
  - Prevents gradient explosion while allowing drift

**Configuration:**
```yaml
config_fit:
  max_grad_norm: 0.5  # Tighter clipping
```

### Parameter Synchronization

- Ensure **correct global parameters** used in proximal term
- Verify parameter shapes match
- Handle parameter compression if used

### Numerical Stability

- Use **double precision** for μ calculations
- Check for NaN/Inf in proximal term gradients
- Add epsilon (1e-8) to prevent division by zero

## Recommended Experimental Priority

### 🔴 High Impact, Easy to Implement

1. **Increase Non-IID**: Set `alpha = 0.05-0.1`
2. **More Local Epochs**: Set `local_epochs = 30-50`
3. **Lower Learning Rate**: Set `lr = 0.001-0.005`
4. **Tune μ**: Try fixed `proximal_mu = 0.1-0.5` or adaptive with higher `mu_max = 2.0`

### 🟡 Medium Impact, Moderate Effort

5. **Adaptive μ with faster adaptation**: Lower `smoothing_factor`, reduce `warmup_rounds`
6. **More Rounds**: Run for 100+ rounds
7. **Per-client heterogeneity-aware μ**: Assign μ based on client metrics

### 🟢 Lower Priority, Advanced

8. **Learning rate scheduling**: Implement decay strategies
9. **Per-layer μ**: Different regularization per layer
10. **Better signal normalization**: Improve multi-signal adaptive mu

## Quick Configuration Template

```yaml
# High-impact configuration for FedProx
iid: false
alpha: 0.1  # High heterogeneity

adaptive_mu:
  enabled: true
  mode: 'multi_signal'

multi_signal_mu:
  base_mu: 0.01
  mu_min: 0.001
  mu_max: 2.0           # Higher cap
  smoothing_factor: 0.4  # Faster adaptation
  warmup_rounds: 2       # Start early
  weights:
    gradient_divergence: 0.45  # More weight on divergence
    loss_variance: 0.25
    label_entropy: 0.2
    feature_variance: 0.1

config_fit:
  lr: 0.005             # Lower learning rate
  momentum: 0.95
  local_epochs: 30      # More local training
  max_grad_norm: 0.5    # Tighter clipping
```

## Expected Results

With proper tuning, FedProx should show:
- **2-5% higher accuracy** in highly non-IID settings
- **Lower variance** in per-client accuracy
- **Better convergence** in later rounds
- **More stable** training (less oscillation)

## Notes

- FedProx is most effective when:
  1. Data is **highly heterogeneous** (non-IID)
  2. Clients train **many epochs** locally
  3. There's significant **client drift** in vanilla FedAvg
  
- The benefits become more pronounced over **longer training periods** (100+ rounds)

- Always compare against **well-tuned FedAvg baseline** to ensure fair comparison
