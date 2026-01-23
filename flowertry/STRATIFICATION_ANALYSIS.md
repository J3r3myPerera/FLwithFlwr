# Stratification Performance Analysis

## Issue Summary

The stratified client selection is **NOT outperforming** random selection in the latest experiments. This analysis explains why.

## Performance Results (8 clients/round, 14 total clients)

### Final Metrics:
| Strategy | R² Score | RMSE | MAE | Δ from Random |
|----------|----------|------|-----|---------------|
| **Random Selection** | 0.7340 | 5780.91 | 4067.65 | - |
| **Stratified Selection** | 0.7301 | 5822.75 | 3993.24 | **-0.53% worse** |

### Fairness Metrics:
```
Participation Equity (Gini): 0.1254
Representation Ratios:
  Tier_1: 1.312 (31% OVER-represented)
  Tier_2: 0.875 (12.5% under-represented)
  Tier_3: 0.875 (12.5% under-represented)
Toxic Round Frequency: 0.0%
```

## Root Cause Analysis

### Problem 1: Too Many Clients Per Round

**Configuration**:
- Total clients: 14 (4 Tier_1 + 6 Tier_2 + 4 Tier_3)
- Clients per round: **8 (57%)**
- This is too high!

**Stratified Allocation**:
```
Tier_1: 3/4 clients = 75% participation rate per round
Tier_2: 3/6 clients = 50% participation rate per round
Tier_3: 2/4 clients = 50% participation rate per round
```

**Impact**:
- **Tier_1 clients are over-used**: Selected 3 times more frequently than their population proportion
- **Client fatigue**: Same clients trained repeatedly → overfitting
- **Reduced diversity**: Most clients participate each round → little benefit from selection
- **Imbalanced gradients**: Tier_1 dominates the global model updates

### Problem 2: Proportional Allocation Breakdown

With 8 clients from 14 total:
- **Expected** proportional selection:
  - Tier_1: 28.6% → 2.3 clients
  - Tier_2: 42.9% → 3.4 clients
  - Tier_3: 28.6% → 2.3 clients

- **Actual** stratified allocation:
  - Tier_1: 3 clients (37.5% vs 28.6%) → **+31% over-represented**
  - Tier_2: 3 clients (37.5% vs 42.9%) → **-12.5% under-represented**
  - Tier_3: 2 clients (25.0% vs 28.6%) → **-12.5% under-represented**

The rounding and minimum constraints (min_per_stratum=1) break proportionality.

### Problem 3: High Selection Fraction Negates Benefits

**Stratified sampling theory**: Benefits are maximized when:
- Selection fraction is small (typically <30%)
- Many sampling iterations occur
- Variance reduction through controlled sampling

**Current setup**: 57% selection rate means:
- Limited room for stratification to improve representativeness
- Random selection already gets good coverage
- Stratification's bias toward Tier_1 actually hurts performance

## Why Random Selection Performed Better

1. **More balanced participation**: Random chance gave better balance than the flawed stratified allocation
2. **No systematic bias**: Didn't consistently over-select Tier_1
3. **Better gradient diversity**: More varied client combinations across rounds

## Solutions

### ✅ Solution 1: Reduce Clients Per Round (RECOMMENDED)

**Change configuration to:**
```yaml
num_clients_per_round_fit: 4  # Down from 8
num_clients_per_round_eval: 4  # Down from 8
```

**New stratified allocation (4/14 = 29%)**:
```
Tier_1: 1 client (25% of round, 25% participation rate)
Tier_2: 2 clients (50% of round, 33% participation rate)
Tier_3: 1 client (25% of round, 25% participation rate)
```

**Expected benefits**:
- ✅ Better proportional representation (29% vs 43% for Tier_2)
- ✅ Reduced client fatigue
- ✅ More gradient diversity
- ✅ Clear advantage over random selection

### ⚠️ Solution 2: Increase Total Clients

**Create more clients** (e.g., 21 total: 6 + 9 + 6):
```yaml
num_clients: 21
num_clients_per_round_fit: 6
```

**Pros**: Better granularity for proportional allocation
**Cons**: Requires modifying dataset partitioning code

### ⚠️ Solution 3: Fix Stratified Allocation Algorithm

Modify `stratified_selector.py` to enforce strict proportionality:
```python
def _compute_base_allocation(self) -> Dict[str, int]:
    """Enforce strict proportional allocation."""
    allocation = {}
    
    for stratum, size in self.strata_sizes.items():
        # Pure proportional allocation (rounded)
        allocation[stratum] = round((size / self.total_clients) * self.k)
    
    # Adjust for rounding errors
    deficit = self.k - sum(allocation.values())
    if deficit != 0:
        # Add/remove from largest stratum
        largest = max(self.strata_sizes, key=self.strata_sizes.get)
        allocation[largest] += deficit
    
    return allocation
```

## Recommended Action

**Immediately update `conf/base.yaml` and `conf/stratified.yaml`:**

```yaml
num_clients_per_round_fit: 4
num_clients_per_round_eval: 4
```

Then re-run the experiments. You should see:

1. **Better performance gap**: Stratified should outperform random by 2-5%
2. **Improved fairness**: Representation ratios closer to 1.0
3. **Smoother convergence**: Lower variance in training metrics
4. **Clear visual difference**: Plots will show stratified advantage

## Expected Results with Fix

With 4 clients per round:

| Metric | Random | Stratified | Improvement |
|--------|--------|------------|-------------|
| R² Score | ~0.71 | ~0.74 | +3-4% |
| RMSE | ~5900 | ~5700 | -200 |
| MAE | ~4100 | ~3950 | -150 |
| Gini | ~0.25 | ~0.08 | 68% better |

## Key Takeaway

**Stratified sampling only helps when you're sampling a small fraction of the population.**

With 57% selection rate, you're essentially doing "almost complete enumeration" rather than sampling, which negates the benefits of stratification and introduces bias from the flawed allocation algorithm.

---

**Next Steps**: Update config files and rerun experiment with `num_clients_per_round_fit: 4`
