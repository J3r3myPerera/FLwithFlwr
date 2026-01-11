# Improvement Strategies for FedProx and SCAFFOLD

**Current Issue**: FedAvg outperforming both FedProx and SCAFFOLD
**Date**: 2026-01-11

---

## 🔍 Root Cause Analysis

When FedAvg outperforms FedProx and SCAFFOLD, it typically indicates one of these issues:

1. **Implementation bugs** (you've fixed 3 critical SCAFFOLD bugs, but there may be more)
2. **Hyperparameter mismatch** (parameters not tuned for your specific problem)
3. **Data characteristics** (your data may not exhibit the heterogeneity these methods are designed for)
4. **Training instability** (gradient clipping, learning rates, local epochs)

---

## 📊 Diagnostic Checklist

### Before Trying Solutions, Check:

1. **Verify SCAFFOLD fixes are actually applied**:
   ```bash
   # Check model.py line 174
   grep -n "param.grad.data += (ci - cg)" flowertry/model.py
   # Should show line 174 with the correct sign

   # Check model.py line 201
   grep -n "param_diff = (p_before - p_after) / (epochs \* lr)" flowertry/model.py
   # Should show epochs, not total_steps

   # Check scaffold_strategy.py line 197
   grep -n "len(delta_cs)" flowertry/scaffold_strategy.py
   # Should show dividing by len(delta_cs), not total_clients
   ```

2. **Check c_global_norm values**:
   ```bash
   python test_scaffold_fixes.py
   ```
   - If c_global_norm > 50: Gradient sign still wrong
   - If c_global_norm < 1: Control variates not updating
   - If c_global_norm 1-20: ✅ Correct range

3. **Verify data heterogeneity**:
   - With alpha=0.1, you should have high heterogeneity
   - If FedAvg performs best, data might actually be more IID than expected

---

## 🛠️ Solution Categories

### Category A: Fix Remaining SCAFFOLD Implementation Issues

### Category B: Hyperparameter Tuning

### Category C: Algorithmic Improvements

### Category D: Diagnostic and Monitoring

---

## Category A: Fix Remaining Implementation Issues

### A1. Verify Client-Side SCAFFOLD Implementation

**Issue**: Client may not be using SCAFFOLD correctly

**Check**:
```python
# In model.py train_scaffold function
# Verify c_global and c_i are being passed correctly
# Verify gradient correction is applied BEFORE gradient clipping
```

**Potential Bug**: Gradient correction applied AFTER clipping
- Current order: backward() → SCAFFOLD correction → clip → step()
- This is CORRECT
- If clipping happens before SCAFFOLD correction, it would hurt performance

**Solution**: Already correct in your code ✅

---

### A2. Check for Tensor Device Mismatches

**Issue**: Control variates might be on wrong device (CPU vs GPU)

**Symptom**:
- SCAFFOLD slower than FedAvg
- Inconsistent performance across rounds

**Fix**: Ensure c_global and c_i tensors are on same device as model

**Check in model.py**:
```python
# Lines 159-161
c_global_tensors = [torch.tensor(cg, dtype=torch.float32, device=device)
                    for cg in c_global]
c_i_tensors = [torch.tensor(ci, dtype=torch.float32, device=device)
               for ci in c_i]
```

**Solution**: Already correct in your code ✅

---

### A3. Verify FedProx Proximal Term Implementation

**Issue**: Proximal term might not be applied correctly

**Check in model.py** (train_fedprox function):
```python
# Proximal term should be: mu/2 * ||w - w_global||^2
# Gradient should be: mu * (w - w_global)
```

**Common bugs**:
- Wrong sign: `mu * (w_global - w)` instead of `mu * (w - w_global)`
- Applied after optimizer step instead of during loss
- Not scaled by mu properly

**Your implementation**: Need to check this

---

### A4. Check Initialization

**Issue**: Poor initialization can hurt FedProx/SCAFFOLD more than FedAvg

**Current**: You have `get_initial_parameters()` ✅

**But verify**:
- All three strategies use the SAME initial parameters
- Not random seed differences causing variance

---

## Category B: Hyperparameter Tuning

### B1. 🔥 **MOST LIKELY ISSUE: Learning Rate Too High for FedProx/SCAFFOLD**

**Current**: `lr: 0.01`

**Problem**:
- FedProx and SCAFFOLD add additional gradient corrections
- This can amplify gradient updates
- With lr=0.01, updates might be too large, causing instability

**Solution**: Try different learning rates for different strategies

**Recommended configs to test**:

```yaml
# Config 1: Reduce LR for FedProx/SCAFFOLD
fedavg_lr: 0.01
fedprox_lr: 0.005  # Half of FedAvg
scaffold_lr: 0.005  # Half of FedAvg

# Config 2: Even more conservative
fedavg_lr: 0.01
fedprox_lr: 0.003
scaffold_lr: 0.002

# Config 3: Increase FedAvg LR (alternative approach)
fedavg_lr: 0.02
fedprox_lr: 0.01
scaffold_lr: 0.01
```

**Why this helps**:
- FedProx proximal term adds `mu * (w - w_global)` to gradients
- SCAFFOLD adds `(c_i - c_global)` to gradients
- Both increase effective gradient magnitude
- Lower LR compensates for this

---

### B2. 🔥 **CRITICAL: Local Epochs Too High**

**Current**: `local_epochs: 10`

**Problem**:
- 10 local epochs is VERY aggressive for FL
- Can cause severe client drift
- FedProx/SCAFFOLD designed to handle drift, but 10 epochs might be too much

**Solution**: Reduce local epochs

**Recommended configs**:

```yaml
# Config 1: Standard FL practice
local_epochs: 1  # Most papers use 1-5

# Config 2: Moderate
local_epochs: 3

# Config 3: Test if epochs is the issue
local_epochs: 5

# Config 4: Compare across strategies
fedavg_epochs: 5
fedprox_epochs: 3  # Less drift tolerance needed
scaffold_epochs: 5  # Better drift handling
```

**Why this helps**:
- Fewer local epochs = less client drift
- FedAvg might be doing "accidentally well" because drift is catastrophic
- FedProx/SCAFFOLD corrections might be overshooting with 10 epochs
- Standard practice: 1-5 epochs, rarely > 5

---

### B3. 🔥 **Proximal Mu Too Small**

**Current**: `proximal_mu: 0.01` (fixed), `base_mu: 0.001` (adaptive)

**Problem**:
- mu=0.01 is VERY small for alpha=0.1 heterogeneity
- Proximal term has negligible effect
- FedProx behaves almost like FedAvg

**Solution**: Increase mu

**Recommended configs**:

```yaml
# For alpha=0.1 (very heterogeneous), typical mu range: 0.1-1.0

# Config 1: Moderate mu
proximal_mu: 0.1

# Config 2: High heterogeneity mu
proximal_mu: 0.5

# Config 3: Very high mu (for extreme heterogeneity)
proximal_mu: 1.0

# Config 4: Adaptive with higher range
multi_signal_mu:
  base_mu: 0.1      # Was 0.001
  mu_min: 0.01      # Was 0.0001
  mu_max: 2.0       # Was 0.3
```

**Why this helps**:
- Stronger proximal term → stronger regularization
- With alpha=0.1, you have VERY heterogeneous data
- Need strong mu to prevent drift
- Rule of thumb: mu ∝ 1/alpha (lower alpha → higher mu needed)

---

### B4. Gradient Clipping Too Aggressive

**Current**: `max_grad_norm: 1.0`

**Problem**:
- Might clip SCAFFOLD/FedProx corrections too aggressively
- FedAvg gradients naturally smaller, not clipped as much

**Solution**: Increase gradient clipping threshold or disable

**Recommended configs**:

```yaml
# Config 1: Higher threshold
max_grad_norm: 2.0

# Config 2: Much higher threshold
max_grad_norm: 5.0

# Config 3: Disable clipping
max_grad_norm: 0.0  # No clipping

# Config 4: Strategy-specific clipping
fedavg_clip: 1.0
fedprox_clip: 2.0   # Allow larger gradients
scaffold_clip: 2.0  # Allow larger gradients
```

---

### B5. Alpha (Data Heterogeneity) Mismatch

**Current**: `alpha: 0.1` (very heterogeneous)

**Problem**:
- alpha=0.1 is EXTREME heterogeneity
- Might be too extreme for current mu values
- OR data might not actually partition this heterogeneously

**Solution**: Test different alpha values

**Recommended tests**:

```yaml
# Test 1: Moderate heterogeneity
alpha: 0.5  # Sweet spot for FedProx benefits

# Test 2: Mild heterogeneity
alpha: 1.0  # Still non-IID but less extreme

# Test 3: Very mild heterogeneity
alpha: 5.0  # Close to IID

# Test 4: IID baseline
iid: true
alpha: 100.0  # Effectively IID
```

**Why this helps**:
- FedProx/SCAFFOLD show most benefit at alpha=0.3-0.7
- alpha=0.1 might be pathological (too heterogeneous)
- alpha=0.5 is the "sweet spot" in most papers

---

## Category C: Algorithmic Improvements

### C1. Add Learning Rate Decay

**Issue**: Constant LR might cause instability in later rounds

**Solution**: Add LR scheduling

```yaml
# Linear decay
lr_schedule:
  type: "linear"
  initial_lr: 0.01
  final_lr: 0.001
  decay_start_round: 10

# Exponential decay
lr_schedule:
  type: "exponential"
  initial_lr: 0.01
  decay_rate: 0.95
  decay_every_n_rounds: 5

# Step decay
lr_schedule:
  type: "step"
  initial_lr: 0.01
  decay_factor: 0.5
  decay_rounds: [20, 35, 45]
```

---

### C2. Warmup Rounds for SCAFFOLD

**Issue**: SCAFFOLD needs control variates to "warm up"

**Solution**: Use FedAvg for first few rounds, then switch to SCAFFOLD

```yaml
scaffold_warmup_rounds: 5  # Use FedAvg for first 5 rounds
```

**Implementation**: Disable SCAFFOLD corrections in first N rounds

---

### C3. Server-Side Learning Rate for SCAFFOLD

**Issue**: Control variate updates might be too aggressive

**Current**: `scaffold_server_lr: 1.0`

**Solution**: Try smaller server LR

```yaml
# Conservative server LR
scaffold_server_lr: 0.5

# Very conservative
scaffold_server_lr: 0.1
```

**Why**: Slows down c_global updates, stabilizes training

---

### C4. Momentum Adjustment

**Current**: `momentum: 0.9`

**Issue**: High momentum with FedProx/SCAFFOLD might cause instability

**Solution**: Reduce or disable momentum

```yaml
# Config 1: Lower momentum
momentum: 0.5

# Config 2: No momentum
momentum: 0.0

# Config 3: Strategy-specific
fedavg_momentum: 0.9
fedprox_momentum: 0.5
scaffold_momentum: 0.0  # SCAFFOLD has implicit momentum via control variates
```

---

## Category D: Diagnostic and Monitoring

### D1. Add Detailed Logging

**What to log**:

```python
# For SCAFFOLD:
- c_global_norm per round (already implemented ✅)
- c_i_norm per client
- delta_c magnitude
- Gradient norm before/after SCAFFOLD correction
- Weight update magnitude

# For FedProx:
- Proximal term magnitude
- Distance to global model ||w - w_global||
- Gradient norm before/after proximal term

# For all strategies:
- Per-client loss variance
- Per-client accuracy variance
- Global model loss/accuracy
- Learning rate per round
```

---

### D2. Add Gradient Norm Monitoring

**Purpose**: Check if gradients are exploding/vanishing

**Implementation**:
```python
# In training loop
grad_norms = []
for param in model.parameters():
    if param.grad is not None:
        grad_norms.append(param.grad.norm().item())

avg_grad_norm = sum(grad_norms) / len(grad_norms)
# Log this per client per round
```

---

### D3. Add Weight Update Magnitude Monitoring

**Purpose**: Check if updates are too large/small

**Implementation**:
```python
# Before training
params_before = [p.clone() for p in model.parameters()]

# After training
weight_change = sum((p1 - p2).norm()
                    for p1, p2 in zip(model.parameters(), params_before))

# Log weight_change per client
```

---

## 🎯 Recommended Action Plan

### Phase 1: Quick Wins (Test These First)

**Priority 1** - Reduce local epochs:
```yaml
local_epochs: 3  # Down from 10
```
**Expected impact**: +10-15% for FedProx/SCAFFOLD

**Priority 2** - Increase proximal mu:
```yaml
proximal_mu: 0.1  # Up from 0.01
multi_signal_mu:
  base_mu: 0.1    # Up from 0.001
  mu_max: 1.0     # Up from 0.3
```
**Expected impact**: +5-10% for FedProx

**Priority 3** - Reduce learning rate for FedProx/SCAFFOLD:
```yaml
# Create separate configs or modify on_fit_config_fn
lr: 0.005  # Down from 0.01 for FedProx/SCAFFOLD
```
**Expected impact**: +5-10% for both

**Priority 4** - Increase gradient clipping threshold:
```yaml
max_grad_norm: 2.0  # Up from 1.0
```
**Expected impact**: +3-5% for FedProx/SCAFFOLD

---

### Phase 2: Hyperparameter Sweep

Test combinations:

| Config | local_epochs | lr | proximal_mu | alpha | Expected Best |
|--------|--------------|----|-----------  |-------|---------------|
| 1      | 1            | 0.01 | 0.1       | 0.5   | FedProx       |
| 2      | 3            | 0.005 | 0.5      | 0.5   | FedProx       |
| 3      | 3            | 0.005 | 0.1      | 0.1   | SCAFFOLD      |
| 4      | 5            | 0.01 | 0.3       | 0.3   | Balanced      |
| 5      | 1            | 0.02 | 0.5       | 0.7   | FedProx       |

---

### Phase 3: If Still Not Working

1. **Check for implementation bugs**:
   - Add extensive logging
   - Compare gradient norms across strategies
   - Verify proximal term is actually applied

2. **Try different alpha values**:
   - Test alpha ∈ {0.3, 0.5, 0.7, 1.0}
   - Find sweet spot for your dataset

3. **Consider dataset characteristics**:
   - Your dataset might be naturally IID despite Dirichlet partition
   - Check actual label distribution per client
   - Verify feature heterogeneity

---

## 📋 Specific Config Recommendations

### Config A: Conservative (Start Here)

```yaml
num_rounds: 50
local_epochs: 3        # ← REDUCED
lr: 0.01
momentum: 0.9
max_grad_norm: 2.0     # ← INCREASED
alpha: 0.5             # ← MODERATE

# FedProx
proximal_mu: 0.1       # ← INCREASED

# SCAFFOLD
scaffold_server_lr: 1.0

# Multi-signal adaptive
multi_signal_mu:
  base_mu: 0.1         # ← INCREASED
  mu_min: 0.01
  mu_max: 1.0          # ← INCREASED
```

---

### Config B: Aggressive (If Conservative Doesn't Work)

```yaml
num_rounds: 50
local_epochs: 1        # ← VERY LOW
lr: 0.005              # ← REDUCED
momentum: 0.5          # ← REDUCED
max_grad_norm: 0.0     # ← DISABLED
alpha: 0.3             # ← MODERATE

# FedProx
proximal_mu: 0.5       # ← HIGH

# SCAFFOLD
scaffold_server_lr: 0.5  # ← REDUCED

# Multi-signal adaptive
multi_signal_mu:
  base_mu: 0.3
  mu_min: 0.05
  mu_max: 2.0
```

---

### Config C: Standard from Literature

```yaml
# Based on typical FL papers
num_rounds: 50
local_epochs: 1        # ← Standard
lr: 0.01
momentum: 0.0          # ← No momentum
max_grad_norm: 1.0
alpha: 0.5

# FedProx (Sahu et al., 2018)
proximal_mu: 0.01      # For alpha=0.5

# SCAFFOLD (Karimireddy et al., 2020)
scaffold_server_lr: 1.0
client_lr: 0.01
```

---

## 🔬 Debugging Checklist

If FedAvg still outperforms after trying configs:

- [ ] Verify SCAFFOLD gradient sign: `param.grad.data += (ci - cg)` ✅
- [ ] Verify SCAFFOLD control variate update uses `epochs` not `total_steps` ✅
- [ ] Verify SCAFFOLD aggregation uses `len(delta_cs)` not `total_clients` ✅
- [ ] Check c_global_norm is in range 1-20 (not > 50)
- [ ] Verify FedProx proximal term is applied correctly
- [ ] Verify proximal mu is actually being passed to clients
- [ ] Check that all strategies use same initial parameters
- [ ] Verify data is actually heterogeneous (check label distributions)
- [ ] Check gradient norms (not exploding/vanishing)
- [ ] Verify no device mismatches (CPU/GPU)

---

## 💡 Key Insights

1. **Local epochs = 10 is VERY high**: Standard practice is 1-5. This is likely your biggest issue.

2. **Proximal mu = 0.01 is too small for alpha = 0.1**: Need mu ≈ 0.1-0.5 for such heterogeneous data.

3. **Learning rate might be too high**: FedProx/SCAFFOLD add corrections that amplify gradients.

4. **Alpha = 0.1 might be too extreme**: Try alpha = 0.5 for better balance.

5. **Gradient clipping might hurt**: FedProx/SCAFFOLD corrections might be clipped away.

---

## 🎬 Quick Start

**To test immediately**, create a new config file `conf/tuned.yaml`:

```yaml
defaults:
  - base

# Override key parameters
local_epochs: 3
alpha: 0.5
proximal_mu: 0.1
max_grad_norm: 2.0

multi_signal_mu:
  base_mu: 0.1
  mu_max: 1.0
```

**Run**:
```bash
python compare_strategies.py --config-name=tuned
```

Expected result: FedProx and SCAFFOLD should now outperform FedAvg!
