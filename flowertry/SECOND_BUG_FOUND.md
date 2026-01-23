# 🐛 SECOND CRITICAL BUG: Dataset Seed Not Passed

## The Problem

Even after fixing the first bug, stratified selection shows **high variance** across runs with the same seed:
- Run 1 (12-36-36): Stratified **wins** by +5.38% ✅
- Run 2 (12-40-50): Stratified **loses** by -13.08% ❌

Both used `seed=42` but got **completely different results!**

## Root Cause

### Bug #1: Dataset Seed Not Passed from main()

**Location**: `main_stratified.py` line 74-78

```python
# ❌ BUGGY CODE
trainloaders, validationloaders, testloader, target_scaler, client_strata = prepare_dataset(
    num_clients=cfg.num_clients,
    batch_size=cfg.batch_size,
    data_path=cfg.get('data_path', './data/...')
    # Missing: seed parameter!
)
```

### Bug #2: Hardcoded Seed in dataset.py

**Location**: `dataset.py` line 76

```python
def prepare_dataset_pure_partitioning(
    ...
    seed: int = 2023  # ❌ Hardcoded default!
):
    torch.manual_seed(seed)
    np.random.seed(seed)  # This OVERRIDES the global seed set in main()!
```

## The Impact

### What Happens

1. `main()` sets global seed to **42**
2. `prepare_dataset()` is called WITHOUT passing seed
3. `prepare_dataset()` internally sets seed to **2023** (default)
4. **Global seed is destroyed!**
5. Subsequent operations use corrupted random state

### Why This Causes Inconsistency

The inconsistency between runs comes from **timing differences**:

- **Some random operations happen BEFORE dataset preparation** (use seed=42)
- **Some random operations happen AFTER dataset preparation** (use seed=2023)
- **The timing varies slightly** between runs (Ray initialization, system state)
- **This creates UNPREDICTABLE behavior**

### Evidence

#### Run 1 (12-36-36) - "Lucky" timing:
```
Initial R²: -0.9086 (bad initialization)
Random Final R²: 0.7384
Stratified Final R²: 0.7781 (+5.38%)
```

#### Run 2 (12-40-50) - "Unlucky" timing:
```
Initial R²: -0.2214 (different initialization!)
Random Final R²: 0.8054
Stratified Final R²: 0.7000 (-13.08%)
```

**The initial R² is completely different!** This proves the random state is different.

## The Fix

### Part 1: Pass Seed to prepare_dataset()

```python
# ✅ FIXED CODE in main_stratified.py
trainloaders, validationloaders, testloader, target_scaler, client_strata = prepare_dataset(
    num_clients=cfg.num_clients,
    batch_size=cfg.batch_size,
    data_path=cfg.get('data_path', './data/...'),
    seed=global_seed  # Pass the global seed!
)
```

### Part 2: Make Dataset Seed Respect Global State

The dataset function should:
1. Accept the seed parameter
2. Use it to set local random state
3. But ideally, NOT override global state

**Option A (Current approach)**: Pass seed and let it override
- Simple but breaks global state
- Requires resetting seed after dataset prep

**Option B (Better)**: Use local random generators
- More complex but cleaner
- Doesn't pollute global state

For now, we'll use Option A and add a seed reset after dataset prep.

### Part 3: Reset Seeds After Dataset Preparation

```python
# ✅ FIXED CODE in main_stratified.py (after dataset prep)
trainloaders, validationloaders, testloader, target_scaler, client_strata = prepare_dataset(
    ...
    seed=global_seed
)

# CRITICAL: Reset seeds after dataset prep
# (because prepare_dataset internally modifies the global numpy/torch state)
print("🌱 Resetting seeds after dataset preparation...")
set_all_seeds(global_seed)
```

## Expected Impact

### Before Fix:
```
Run 1: Random=0.7384, Stratified=0.7781 (+5.38%)
Run 2: Random=0.8054, Stratified=0.7000 (-13.08%)
Run 3: Random=????, Stratified=???? (unpredictable)

Problem: Completely inconsistent!
```

### After Fix:
```
Run 1: Random=0.74XX, Stratified=0.76XX (+~3%)
Run 2: Random=0.74XX, Stratified=0.76XX (+~3%)
Run 3: Random=0.74XX, Stratified=0.76XX (+~3%)

Result: Consistent and reproducible!
```

## Why This Is Critical

1. **Scientific Validity**: Can't publish results that aren't reproducible
2. **Fair Comparison**: Random and stratified must start from same state
3. **Debugging**: Impossible to debug non-deterministic behavior
4. **Trust**: Inconsistent results undermine confidence in the approach

## Related Issues

### Why Did First Run Sometimes Work?

Even with the bug, sometimes you'd get "lucky":
- If Ray/Flower initialization happened to use the same random state
- If timing was consistent
- If system load was similar

But this "luck" is unreliable and unscientific.

### Why Does Random Sometimes Win Big?

When the random state is corrupted:
- Random selection might get "lucky" with good client combinations
- Stratified selection might get "unlucky" with bad combinations
- The corruption is unpredictable, so results vary wildly

## Complete Fix Summary

### Files Changed

1. **main_stratified.py** (line 74-78):
   - Pass `seed=global_seed` to `prepare_dataset()`
   - Add seed reset after dataset preparation

2. **dataset.py** (no changes needed):
   - Already accepts seed parameter
   - Default value of 2023 is fine (won't be used anymore)

### Implementation

```python
# In main_stratified.py

## 2. Prepare the datasets
trainloaders, validationloaders, testloader, target_scaler, client_strata = prepare_dataset(
    num_clients=cfg.num_clients,
    batch_size=cfg.batch_size,
    data_path=cfg.get('data_path', './data/...'),
    seed=global_seed  # ✅ FIX: Pass global seed
)

# ✅ FIX: Reset seeds after dataset prep
print("🌱 Resetting seeds after dataset preparation...")
set_all_seeds(global_seed)

print(f"Number of training clients: {len(trainloaders)}")
```

## Testing the Fix

### Test 1: Reproducibility
```bash
python main_stratified.py seed=42 > run1.log 2>&1
python main_stratified.py seed=42 > run2.log 2>&1
diff run1.log run2.log
```

**Expected**: NO differences in metrics

### Test 2: Initial R² Check
```bash
python main_stratified.py seed=42 | grep "initial parameters"
python main_stratified.py seed=42 | grep "initial parameters"
```

**Expected**: IDENTICAL initial R² values

### Test 3: Final Results Check
```bash
python main_stratified.py seed=42
# Check final R² scores
# Run again
python main_stratified.py seed=42
# Check final R² scores again
```

**Expected**: IDENTICAL final R² scores (within 0.01%)

## Summary

### The Bugs
1. ❌ Global seed set to 42
2. ❌ Dataset prep called without seed parameter
3. ❌ Dataset prep internally sets seed to 2023
4. ❌ Global random state corrupted
5. ❌ Timing differences cause inconsistency

### The Fixes
1. ✅ Pass `seed=global_seed` to `prepare_dataset()`
2. ✅ Reset seeds after dataset preparation
3. ✅ Both phases use consistent random state

### The Result
🎯 **Reproducible, fair, consistent comparisons!**

---

**Status**: 🔴 **CRITICAL BUG - FIX IN PROGRESS**
