# Client Selection Strategy for Enhanced Hybrid FedProx-SCAFFOLD

## Overview

This document describes the quality-based client selection strategy added to the Enhanced Hybrid FedProx-SCAFFOLD approach. Instead of randomly sampling clients each round, we select clients based on their contribution quality using three performance-based metrics.

## Motivation

Random client selection has limitations:

- **Equal treatment**: All clients are sampled with equal probability regardless of their contribution quality
- **Noisy clients**: Clients with poor data quality or high drift get selected as often as high-quality clients
- **Slow convergence**: Low-quality updates slow down global model improvement

Quality-based selection addresses these issues by prioritizing clients that:

1. Achieve better local training loss (relative to their data difficulty)
2. Provide gradients aligned with global improvement direction
3. Have historically contributed to accuracy improvements

## Client Quality Metrics

### 1. Local Loss Quality (Q_loss)

**Formula:**

```
Q_loss(i) = 1 / (1 + exp(loss_i - loss_median))
```

**Description:**

- Sigmoid-normalized score comparing client's loss to the median across all clients
- **Higher is better** (scores in range 0 to 1)
- Clients with lower loss relative to data difficulty receive higher scores
- Very low loss may indicate overfitting, which the sigmoid naturally dampens

**Intuition:**
A client that achieves low training loss (given their potentially difficult local data distribution) provides more reliable gradient updates.

**Weight:** 0.3 (configurable via `quality_loss_weight`)

---

### 2. Gradient Utility Score (Q_grad)

**Formula:**

```
Q_grad(i) = max(0, cos(g_i, g_global))
```

**Description:**

- Cosine similarity between client's gradient and global update direction
- **Higher is better** (scores in range 0 to 1)
- Measures how much the client's gradient contributes to global model improvement
- Negative similarities are clipped to 0 (opposite direction = no utility)

**Intuition:**
A client whose gradient points in the same direction as the global optimization trajectory is actively helping the model improve, not fighting against it.

**Weight:** 0.4 (configurable via `quality_grad_weight`)

---

### 3. Historical Accuracy Contribution (Q_acc)

**Formula:**

```
Q_acc(i) = EMA(Δacc | client i participated)
```

**Description:**

- Exponential moving average of accuracy deltas when this client participates
- Tracks each client's historical impact on global test accuracy
- Can be positive (accuracy improver) or negative (accuracy degrader)
- Normalized to [0, 1] range for combination with other metrics

**Intuition:**
Clients whose participation correlates with accuracy improvements should be selected more often. This captures long-term contribution patterns beyond immediate gradient quality.

**Weight:** 0.3 (configurable via `quality_acc_weight`)

**EMA Update:**

```python
Q_acc(i) = α * Δacc_current + (1 - α) * Q_acc_previous
```

where α = 0.5 (configurable via `quality_alpha`)

---

## Combined Quality Score

The overall client quality score is a weighted combination:

```
Q_total(i) = w1·Q_loss(i) + w2·Q_grad(i) + w3·Q_acc(i)
```

**Default weights:**

- w1 = 0.3 (loss quality)
- w2 = 0.4 (gradient utility)
- w3 = 0.3 (accuracy contribution)

These weights sum to 1.0 and can be adjusted based on domain requirements.

---

## Selection Algorithm

### Hybrid Top-K + Probabilistic Sampling

To balance exploitation (selecting best clients) with exploration (maintaining diversity), we use a hybrid approach:

```python
def select_clients(num_clients):
    # Step 1: Sort all clients by quality score (descending)
    clients_sorted = sort_by_quality(all_clients)

    # Step 2: Deterministically select top 50%
    top_k = num_clients // 2
    selected = clients_sorted[:top_k]

    # Step 3: Probabilistically sample remaining 50%
    remaining = clients_sorted[top_k:]
    probs = quality_scores / sum(quality_scores)
    sampled = random_sample(remaining, size=num_clients-top_k, probs=probs)

    return selected + sampled
```

**Rationale:**

- **Top-K selection (50%)**: Ensures consistently high-quality clients are included
- **Probabilistic sampling (50%)**: Maintains exploration and prevents over-reliance on few clients
- **Weighted probabilities**: Better clients in the remaining pool have higher selection chance

**Example with 10 clients to select:**

- Top 5: Highest quality clients (always selected)
- Bottom 5: Sampled from remaining pool with quality-weighted probabilities

---

## Configuration Parameters

All parameters can be configured in `conf/base.yaml` under the `hybrid` section:

```yaml
hybrid:
  # Client Selection Strategy
  use_quality_selection: true # Enable quality-based selection
  quality_alpha: 0.5 # EMA smoothing for Q_acc (0-1)
  quality_loss_weight: 0.3 # Weight for Q_loss
  quality_grad_weight: 0.4 # Weight for Q_grad
  quality_acc_weight: 0.3 # Weight for Q_acc
```

### Parameter Tuning Guidelines

| Parameter               | Range      | Effect                                | Recommendation                    |
| ----------------------- | ---------- | ------------------------------------- | --------------------------------- |
| `use_quality_selection` | true/false | Enable/disable feature                | true for heterogeneous settings   |
| `quality_alpha`         | 0.0-1.0    | EMA smoothing for Q_acc               | 0.5 balances recent vs historical |
| `quality_loss_weight`   | 0.0-1.0    | Importance of local loss              | 0.2-0.4 typical                   |
| `quality_grad_weight`   | 0.0-1.0    | Importance of gradient alignment      | 0.3-0.5 (most direct signal)      |
| `quality_acc_weight`    | 0.0-1.0    | Importance of historical contribution | 0.2-0.4 typical                   |

**Note:** Weights should sum to 1.0 for interpretability.

---

## Implementation Details

### Tracking State

The strategy maintains several dictionaries to track client metrics:

```python
self.client_loss_history: Dict[str, List[float]]           # Loss history per client
self.client_gradient_quality: Dict[str, float]             # Latest Q_grad
self.client_accuracy_contribution: Dict[str, float]        # EMA of Q_acc
self.client_quality_scores: Dict[str, float]               # Combined Q_total
self.last_round_accuracy: Optional[float]                  # For computing Δacc
self.round_loss_stats: Dict[str, float]                    # Per-round loss for median
```

### Computation Flow

**During `aggregate_fit()`:**

1. Collect loss statistics from all clients
2. Compute Q_loss for each client (using loss median)
3. Compute Q_grad for each client (using global update direction)
4. Store metrics temporarily (Q_acc requires evaluation result)

**During `evaluate()`:**

1. Get current global test accuracy
2. Compute Δacc = current_accuracy - last_round_accuracy
3. Update Q_acc for all clients that participated
4. Compute final Q_total combining all three metrics

**During `configure_fit()` (next round):**

1. Use Q_total scores to select high-quality clients
2. Apply hybrid top-k + probabilistic sampling

---

## Expected Benefits

### 1. Faster Convergence

- Higher-quality gradients are aggregated more frequently
- Global model improvement accelerates

### 2. Better Final Accuracy

- Low-quality or adversarial clients are selected less often
- Model benefits from consistently good updates

### 3. Robustness

- Gracefully handles heterogeneous data distributions
- Automatically down-weights clients with poor data or high drift

### 4. Fairness Preservation

- Probabilistic sampling ensures all clients eventually participate
- No permanent exclusion of any client

---

## Experimental Validation

### Baseline Comparison

| Strategy                   | Selection Method | Final Accuracy |
| -------------------------- | ---------------- | -------------- |
| Hybrid (Random)            | Random sampling  | 66.55%         |
| **Hybrid (Quality-Based)** | Quality metrics  | **TBD**        |

_(Results to be updated after testing)_

### Quality Score Evolution

Track how client quality scores evolve over training rounds to verify:

- High-quality clients maintain high scores
- Low-quality clients improve or remain low
- Selection diversity is maintained

---

## Ablation Study

To understand the contribution of each metric:

| Configuration          | Q_loss | Q_grad | Q_acc | Expected Impact                  |
| ---------------------- | ------ | ------ | ----- | -------------------------------- |
| Loss-only              | 1.0    | 0.0    | 0.0   | Focus on low-loss clients        |
| Gradient-only          | 0.0    | 1.0    | 0.0   | Focus on aligned gradients       |
| Accuracy-only          | 0.0    | 0.0    | 1.0   | Focus on historical contributors |
| **Balanced (default)** | 0.3    | 0.4    | 0.3   | All metrics contribute           |

---

## Future Enhancements

1. **Adaptive Weights**: Automatically adjust metric weights based on training phase

   - Early rounds: Emphasize Q_grad (alignment more critical)
   - Later rounds: Emphasize Q_acc (historical performance)

2. **Client Clustering**: Group clients by quality and ensure diversity across clusters

3. **Fairness Constraints**: Add minimum selection frequency guarantees per client

4. **Dynamic Top-K**: Adjust the deterministic/probabilistic split based on quality distribution

---

## References

1. **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
2. **SCAFFOLD**: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for FL" (ICML 2020)
3. **Client Selection**: Nishio & Yonetani, "Client Selection for Federated Learning with Heterogeneous Resources" (IEEE Access 2019)

---

_Last Updated: January 15, 2026_
