# Pure Tier Partitioning Update Summary

## Overview
Updated the codebase to use **pure tier partitioning** where each client receives data from ONLY one City Tier, creating maximum data heterogeneity for testing the stratified client selection approach.

## Key Changes

### 1. Dataset Configuration (`dataset.py`)
- **New Function**: `prepare_dataset_pure_partitioning()`
  - Creates pure tier-based partitioning (no data mixing between tiers)
  - Each client gets data exclusively from one tier
  - Client distribution: **4 Tier_1 + 6 Tier_2 + 4 Tier_3 = 14 clients total**
  
- **Removed**: Alpha parameter (no longer needed for pure partitioning)
- **Added**: Backward compatibility alias (`prepare_dataset = prepare_dataset_pure_partitioning`)

### 2. Strategy Configuration
- **Changed from**: FedProx (with proximal term mu)
- **Changed to**: FedAvg (pure averaging, no proximal term)
- **Comparison**: Random Selection FedAvg vs Stratified Selection FedAvg

### 3. Configuration Files Updated

#### `conf/stratified.yaml`
```yaml
num_clients: 14  # Changed from 12
# Removed: alpha parameter
# Added: Pure tier partitioning description
```

#### `conf/base.yaml`
```yaml
num_clients: 14  # Changed from 12
# Removed: alpha parameter
# Updated: fedprox config to FedAvg (removed mu)
```

### 4. Main Scripts Updated

#### `main_stratified.py`
- Removed alpha parameter handling
- Updated to use pure tier partitioning
- Changed strategy from `StratifiedFedProx` to `StratifiedFedAvg`
- Removed proximal_mu parameters

#### `main.py`
- Removed alpha parameter handling
- Updated to use pure tier partitioning

## Client Distribution Details

### Pure Tier Partitioning
```
Tier_1: 4 clients (IDs: 0, 1, 2, 3)
Tier_2: 6 clients (IDs: 4, 5, 6, 7, 8, 9)
Tier_3: 4 clients (IDs: 10, 11, 12, 13)
```

**Total: 14 clients**

### Data Characteristics
- **Maximum Heterogeneity**: Each client has data from only one tier
- **No Data Mixing**: Unlike the previous alpha-based approach, there's zero overlap
- **Ideal for Testing Stratified Selection**: This setup maximizes the benefit of stratified sampling

## Stratified Selection Configuration

When selecting 6 clients per round with `min_clients_per_stratum=1`:

### Expected Distribution
- **Tier_1**: ~2 clients (4/14 × 6 ≈ 1.7, rounded to 2)
- **Tier_2**: ~3 clients (6/14 × 6 ≈ 2.6, rounded to 3)
- **Tier_3**: ~2 clients (4/14 × 6 ≈ 1.7, rounded to 2)

This ensures balanced representation across all three tiers in every training round.

## Comparison Experiments

### Baseline: Random Selection + FedAvg
- Uniform random sampling of 6 clients per round
- No guarantee of tier representation
- Can select all clients from one or two tiers (toxic combinations)

### Experimental: Stratified Selection + FedAvg
- Proportional sampling across tiers
- Guarantees minimum 1 client per tier
- Balanced representation prevents toxic combinations
- Reduces gradient variance

## Benefits of This Configuration

1. **Maximum Heterogeneity**: Pure tier partitioning creates the most challenging scenario
2. **Clear Comparison**: FedAvg baseline makes it easier to isolate the effect of stratified selection
3. **Realistic Scenario**: Mimics real-world federated learning where clients have distinct data distributions
4. **Fairness Testing**: Ensures all demographic groups (tiers) are represented in training

## Running the Updated Code

```bash
# Run stratified selection comparison
python main_stratified.py

# Or with specific config
python main_stratified.py --config-name=stratified
```

## Expected Outcomes

With pure tier partitioning and stratified selection, you should observe:

1. **More Stable Convergence**: Stratified selection reduces gradient variance
2. **Better Fairness Metrics**: 
   - Lower Gini coefficient (more equitable client participation)
   - Better representation ratios (closer to 1.0 for all tiers)
   - Lower toxic round frequency
3. **Improved Test Performance**: Better generalization due to balanced training

## Files Modified

1. ✅ `dataset.py` - Added pure tier partitioning function with all necessary imports
2. ✅ `stratified_strategy.py` - Changed from FedProx to FedAvg
3. ✅ `main_stratified.py` - Updated to use FedAvg and pure partitioning
4. ✅ `main.py` - Updated to use pure partitioning
5. ✅ `conf/stratified.yaml` - Updated client count and removed alpha
6. ✅ `conf/base.yaml` - Updated client count and changed to FedAvg

## Next Steps

1. Run experiments with the new configuration
2. Compare results between random and stratified selection
3. Analyze fairness metrics and convergence behavior
4. Generate plots showing the benefits of stratified selection

---

**Note**: The pure tier partitioning approach creates maximum data heterogeneity, making it an ideal testbed for demonstrating the effectiveness of stratified client selection in federated learning.
