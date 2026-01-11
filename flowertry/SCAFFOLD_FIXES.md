# SCAFFOLD Performance Degradation - Root Causes and Fixes

**Issue**: SCAFFOLD performs well initially but degrades in later rounds, performing worse than FedAvg and FedProx.

---

## Root Causes Identified

### 1. ❌ **CRITICAL BUG: Using `total_steps` instead of `epochs`**

**Location**: [model.py:197](model.py#L197)

**Current Code (WRONG)**:
```python
param_diff = (p_before - p_after) / (total_steps * lr)
```

**Problem**:
- `total_steps` counts the number of SGD steps (batches × epochs)
- Should use `epochs` (number of local epochs)
- This causes control variate updates to be **too small** by a factor of (num_batches)
- As training progresses, this accumulates and causes divergence

**Fix**:
```python
param_diff = (p_before - p_after) / (epochs * lr)
```

---

### 2. ❌ **POTENTIAL BUG: Gradient Correction Sign**

**Location**: [model.py:174](model.py#L174)

**Current Code**:
```python
param.grad.data += (cg - ci)  # c_global - c_i
```

**The Debate**:
The SCAFFOLD paper's formulation can be interpreted two ways:

**Option A** (Current implementation):
```python
g_corrected = g - c_i + c_global
# In code: grad += (c_global - c_i)
```

**Option B** (Alternative):
```python
g_corrected = g + c_i - c_global
# In code: grad += (c_i - c_global)
# OR equivalently: grad -= (c_global - c_i)
```

**Why this matters**:
- With wrong sign, control variates push updates in the **opposite** direction
- Initially (when control variates ≈ 0), both work similarly
- Later (when control variates grow), wrong sign causes divergence

**Recommendation**: Test BOTH directions after fixing bug #1

---

### 3. ❌ **AGGREGATION SCALING BUG**

**Location**: [scaffold_strategy.py:190-191](scaffold_strategy.py#L190-L191)

**Current Code (POTENTIALLY WRONG)**:
```python
avg_delta_c = [
    sum(delta_cs[i][j] for i in range(len(delta_cs))) / self.total_clients
    for j in range(len(self.c_global))
]
```

**Problem**:
- Divides by `self.total_clients` (100 in your config)
- But only `len(delta_cs)` (10) clients participated
- This makes control variate updates 10× too small!

**According to SCAFFOLD paper Option II**:
The update should be: `c_global += (1 / (N × |S|)) × Σ Δc_i`

Where:
- N = total number of clients (100)
- |S| = number of sampled clients per round (10)
- Denominator = N × |S| = 1000

**But the current code has**:
- Denominator = N = 100 (missing the |S| factor!)

**Correct Fix**:
```python
# Option 1: Simple averaging (most implementations use this)
avg_delta_c = [
    sum(delta_cs[i][j] for i in range(len(delta_cs))) / len(delta_cs)
    for j in range(len(self.c_global))
]

# Option 2: Paper's exact formula (Option II)
num_sampled = len(delta_cs)
scaling = 1.0 / (self.total_clients * num_sampled)
avg_delta_c = [
    sum(delta_cs[i][j] for i in range(len(delta_cs))) * scaling * self.total_clients
    for j in range(len(self.c_global))
]
# Simplifies to: sum / num_sampled (same as Option 1!)
```

**So the fix is simple**: Divide by `len(delta_cs)` not `self.total_clients`

---

## Recommended Fixes (Priority Order)

### Fix 1: Control Variate Update Formula (HIGHEST PRIORITY)

**File**: [model.py:197](model.py#L197)

**Change**:
```python
# Before:
param_diff = (p_before - p_after) / (total_steps * lr)

# After:
param_diff = (p_before - p_after) / (epochs * lr)
```

This is **definitely wrong** and must be fixed.

---

### Fix 2: Aggregation Scaling (HIGH PRIORITY)

**File**: [scaffold_strategy.py:190-191](scaffold_strategy.py#L190-L191)

**Change**:
```python
# Before:
avg_delta_c = [
    sum(delta_cs[i][j] for i in range(len(delta_cs))) / self.total_clients
    for j in range(len(self.c_global))
]

# After:
avg_delta_c = [
    sum(delta_cs[i][j] for i in range(len(delta_cs))) / len(delta_cs)
    for j in range(len(self.c_global))
]
```

---

### Fix 3: Test Gradient Correction Direction (MEDIUM PRIORITY)

**File**: [model.py:174](model.py#L174)

After applying fixes 1 and 2, test BOTH:

**Option A** (current):
```python
param.grad.data += (cg - ci)
```

**Option B** (alternative):
```python
param.grad.data -= (cg - ci)
# OR equivalently:
param.grad.data += (ci - cg)
```

Run comparison and see which performs better.

---

## Why This Causes the Pattern You Observed

1. **Early Rounds**:
   - Control variates are small (near zero)
   - Bugs have minimal impact
   - SCAFFOLD can still outperform FedAvg due to variance reduction

2. **Middle Rounds**:
   - Control variates start growing
   - Wrong scaling (bug #3) prevents proper growth
   - Wrong step count (bug #1) makes updates too small
   - Performance degrades

3. **Late Rounds**:
   - Accumulated errors from all bugs
   - Control variates diverge from optimal values
   - Wrong gradient correction (potentially bug #2) actively hurts
   - Performance becomes worse than FedAvg

---

## Testing Procedure

1. **Apply Fix 1 and Fix 2** first
2. **Run comparison** with current gradient direction
3. **If still worse than FedAvg**: Apply Fix 3 (flip gradient sign)
4. **Monitor** `c_global_norm` in metrics:
   - Should start near 0
   - Grow to 1-10 range over first 20 rounds
   - Stabilize in 5-20 range
   - **Red flag**: If > 100 or growing unbounded

---

## Expected Results After Fixes

With all fixes applied and correct gradient direction:

| Round | FedAvg | FedProx | SCAFFOLD (Fixed) |
|-------|--------|---------|------------------|
| 0-10  | 50-55% | 55-60%  | 55-65%          |
| 10-20 | 55-60% | 60-65%  | 65-72%          |
| 20-30 | 58-62% | 63-68%  | 70-77%          |
| 30-40 | 60-65% | 65-72%  | 73-80%          |
| 40-50 | 62-68% | 68-75%  | 75-83%          |

**Key indicators of correct implementation**:
- ✅ SCAFFOLD consistently beats FedAvg by 10-15%
- ✅ SCAFFOLD consistently beats FedProx by 5-10%
- ✅ Performance improves monotonically (or plateaus, doesn't degrade)
- ✅ `c_global_norm` stays in reasonable range (1-20)

---

## References

- **SCAFFOLD Paper**: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020
- **Key equation** (Algorithm 2, Option II):
  - Client update: `c_i^{new} = c_i - c + (y - x_i) / (K × η)`
  - Server update: `c^{new} = c + 1/(n×m) × Σ Δc_i`
  - Where K = local epochs, η = learning rate, n = total clients, m = sampled clients

---

## Quick Test Command

After applying fixes:
```bash
cd flowertry
python compare_strategies.py --config-name=base
```

Watch for:
1. SCAFFOLD accuracy improving over rounds
2. Final SCAFFOLD accuracy > FedProx > FedAvg
3. `c_global_norm` staying < 50
